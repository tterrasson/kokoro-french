import json
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AlbertConfig

from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder

try:
    from StyleTTS2.Modules.diffusion.diffusion import AudioDiffusionConditional
    from StyleTTS2.Modules.diffusion.modules import StyleTransformer1d, Transformer1d
    from StyleTTS2.Modules.diffusion.sampler import (
        ADPM2Sampler,
        DiffusionSampler,
        KarrasSchedule,
        KDiffusion,
        LogNormalDistribution,
    )
except ImportError:
    AudioDiffusionConditional = StyleTransformer1d = Transformer1d = None
    ADPM2Sampler = DiffusionSampler = KarrasSchedule = KDiffusion = LogNormalDistribution = None


class KModel(torch.nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False
    ):
        super().__init__()
        if repo_id is None:
            repo_id = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
        )
        self.diffusion = self._build_diffusion(config)
        self.diffusion_available = False
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id])
        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            assert hasattr(self, key), key
            if key == 'diffusion' and self.diffusion is None:
                logger.warning("Ignoring diffusion state_dict because diffusion modules are not importable")
                continue
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)
            if key == 'diffusion':
                self.diffusion_available = True

    def _build_diffusion(self, config: Dict) -> Optional[torch.nn.Module]:
        if AudioDiffusionConditional is None:
            return None

        style_dim = int(config['style_dim'])
        diffusion_config = config.get('diffusion') or {}
        transformer_config = diffusion_config.get('transformer') or {}
        dist_config = diffusion_config.get('dist') or {}
        transformer_defaults = {
            'num_layers': 3,
            'num_heads': 8,
            'head_features': 64,
            'multiplier': 2,
        }
        transformer_defaults.update(transformer_config)

        if config.get('multispeaker', True):
            transformer = StyleTransformer1d(
                channels=style_dim * 2,
                context_embedding_features=self.bert.config.hidden_size,
                context_features=style_dim * 2,
                **transformer_defaults,
            )
        else:
            transformer = Transformer1d(
                channels=style_dim * 2,
                context_embedding_features=self.bert.config.hidden_size,
                **transformer_defaults,
            )

        diffusion = AudioDiffusionConditional(
            in_channels=1,
            embedding_max_length=self.bert.config.max_position_embeddings,
            embedding_features=self.bert.config.hidden_size,
            embedding_mask_proba=float(diffusion_config.get('embedding_mask_proba', 0.1)),
            channels=style_dim * 2,
            context_features=style_dim * 2,
        )
        diffusion.diffusion = KDiffusion(
            net=transformer,
            sigma_distribution=LogNormalDistribution(
                mean=float(dist_config.get('mean', -3.0)),
                std=float(dist_config.get('std', 1.0)),
            ),
            sigma_data=float(dist_config.get('sigma_data', 0.2)),
            dynamic_threshold=0.0,
        )
        diffusion.diffusion.net = transformer
        diffusion.unet = transformer
        return diffusion

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        diffusion_steps: int = 0,
        diffusion_seed: Optional[int] = None,
        diffusion_alpha: float = 0.1,
        diffusion_beta: float = 0.5,
        embedding_scale: float = 1.0,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        input_lengths = torch.full(
            (input_ids.shape[0],), 
            input_ids.shape[-1], 
            device=input_ids.device,
            dtype=torch.long
        )

        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        ref_s = self.sample_diffusion_style(
            bert_dur,
            ref_s,
            steps=diffusion_steps,
            seed=diffusion_seed,
            alpha=diffusion_alpha,
            beta=diffusion_beta,
            embedding_scale=embedding_scale,
        )
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    @torch.no_grad()
    def sample_diffusion_style(
        self,
        bert_dur: torch.FloatTensor,
        ref_s: torch.FloatTensor,
        steps: int = 0,
        seed: Optional[int] = None,
        alpha: float = 0.1,
        beta: float = 0.5,
        embedding_scale: float = 1.0,
    ) -> torch.FloatTensor:
        if steps <= 0:
            return ref_s
        if self.diffusion is None or not self.diffusion_available:
            raise RuntimeError("This KModel was not loaded with trained diffusion weights")

        alpha = float(max(0.0, min(1.0, alpha)))
        beta = float(max(0.0, min(1.0, beta)))
        if alpha == 0.0 and beta == 0.0:
            return ref_s

        generator = None
        if seed is not None:
            generator = torch.Generator(device=ref_s.device)
            generator.manual_seed(int(seed))

        noise = torch.randn(
            ref_s.shape[0],
            1,
            ref_s.shape[1],
            device=ref_s.device,
            dtype=ref_s.dtype,
            generator=generator,
        )
        sampler = DiffusionSampler(
            self.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )
        sampled = sampler(
            noise=noise,
            embedding=bert_dur,
            embedding_scale=float(embedding_scale),
            features=ref_s,
            embedding_mask_proba=0.0,
            num_steps=int(steps),
        ).squeeze(1)

        blended = ref_s.clone()
        blended[:, :128] = (1.0 - alpha) * ref_s[:, :128] + alpha * sampled[:, :128]
        blended[:, 128:] = (1.0 - beta) * ref_s[:, 128:] + beta * sampled[:, 128:]
        return blended

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False,
        diffusion_steps: int = 0,
        diffusion_seed: Optional[int] = None,
        diffusion_alpha: float = 0.1,
        diffusion_beta: float = 0.5,
        embedding_scale: float = 1.0,
    ) -> Union['KModel.Output', torch.FloatTensor]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(
            input_ids,
            ref_s,
            speed,
            diffusion_steps=diffusion_steps,
            diffusion_seed=diffusion_seed,
            diffusion_alpha=diffusion_alpha,
            diffusion_beta=diffusion_beta,
            embedding_scale=embedding_scale,
        )
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform, duration
