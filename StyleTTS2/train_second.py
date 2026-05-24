import copy
import logging
import os
import os.path as osp
import shutil
import time
import traceback
import warnings
from logging import StreamHandler

import click
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from compile_utils import compile_model_for_training
from kokoro_symbols import TextCleaner
from kokoro_tb_utils import extract_voicepack, prepare_test_tokens, run_kokoro_inference
from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, WavLMLoss
from meldataset import build_dataloader
from models import build_model, load_ASR_models, load_checkpoint, load_F0_models
from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from Modules.slmadv import SLMAdversarialLoss
from munch import Munch
from optimizers import build_optimizer
from parallel_utils import MyDataParallel
from progress_utils import make_progress_bar, metric_postfix, metric_value
from torch.utils.tensorboard.writer import SummaryWriter
from utils import (
    get_data_path_list,
    length_to_mask,
    log_norm,
    mask_from_lens,
    maximum_path,
    recursive_munch,
)
from Utils.PLBERT.util import load_plbert

if getattr(torch, "_original_load", None) is None:
    torch._original_load = torch.load
    torch.load = lambda *args, **kwargs: torch._original_load(
        *args, **{**kwargs, "weights_only": False}
    )

warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def linear_warmup(epoch, start_epoch, warmup_epochs, start_value=0.0, end_value=1.0):
    if epoch < start_epoch:
        return start_value
    if warmup_epochs <= 0:
        return end_value
    progress = min(1.0, max(0.0, (epoch - start_epoch + 1) / warmup_epochs))
    return start_value + (end_value - start_value) * progress


@click.command()
@click.option("-p", "--config_path", default="Configs/config.yml", type=str)
@click.option("-n", "--run_name", default=None, type=str, help="Run name for TensorBoard (defaults to timestamp)")
def main(config_path, run_name):
    config = yaml.safe_load(open(config_path))

    log_dir = config["log_dir"]
    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    if run_name is None:
        run_name = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(osp.join(log_dir, "tensorboard", run_name))

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, "train.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(levelname)s:%(asctime)s: %(message)s")
    )
    logger.addHandler(file_handler)

    batch_size = config.get("batch_size", 10)
    grad_accum = config.get("grad_accum", 1)

    epochs = config.get("epochs_2nd", 200)
    log_interval = config.get("log_interval", 10)
    saving_epoch = config.get("save_freq", 2)

    data_params = config.get("data_params", None)
    sr = config["preprocess_params"].get("sr", 24000)
    train_path = data_params["train_data"]
    val_path = data_params["val_data"]
    root_path = data_params["root_path"]
    min_length = data_params["min_length"]
    OOD_data = data_params["OOD_data"]
    num_workers = data_params.get("num_workers", 2)
    mel_cache_dir = data_params.get("mel_cache_dir", None)

    max_len = config.get("max_len", 200)

    loss_params = Munch(config["loss_params"])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    adv_warmup_epochs = int(getattr(loss_params, "adv_warmup_epochs", 3))
    slmadv_warmup_epochs = int(getattr(loss_params, "slmadv_warmup_epochs", 3))
    real_audio_mix_warmup_epochs = int(getattr(loss_params, "real_audio_mix_warmup_epochs", adv_warmup_epochs))
    lambda_gen_start = float(getattr(loss_params, "lambda_gen_start", 0.05))
    lambda_slm_start = float(getattr(loss_params, "lambda_slm_start", 0.05))

    optimizer_params = Munch(config["optimizer_params"])

    train_list, val_list = get_data_path_list(train_path, val_path)
    device = "cuda"

    train_dataloader = build_dataloader(
        train_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_config={},
        device=device,
        mel_cache_dir=mel_cache_dir,
    )

    val_dataloader = build_dataloader(
        val_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        validation=True,
        num_workers=0,
        device=device,
        dataset_config={},
        mel_cache_dir=mel_cache_dir,
    )

    # load pretrained ASR model
    ASR_config = config.get("ASR_config", False)
    ASR_path = config.get("ASR_path", False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get("F0_path", False)
    pitch_extractor = load_F0_models(F0_path)

    # load PL-BERT model
    BERT_path = config.get("PLBERT_dir", False)
    plbert = load_plbert(BERT_path)

    # build model
    model_params = recursive_munch(config["model_params"])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    [model[key].to(device) for key in model]

    start_epoch = 0
    iters = 0

    load_pretrained = config.get("pretrained_model", "") != "" and config.get(
        "second_stage_load_pretrained", False
    )

    if not load_pretrained:
        if config.get("first_stage_path", "") != "":
            first_stage_path = osp.join(
                log_dir, config.get("first_stage_path", "first_stage.pth")
            )
            print("Loading the first stage model at %s ..." % first_stage_path)
            model, _, start_epoch, iters = load_checkpoint(
                model,
                None,
                first_stage_path,
                load_only_params=True,
                ignore_modules=[
                    "predictor_encoder",
                    "msd",
                    "mpd",
                    "msstft",
                    "subband",
                    "wd",
                    "diffusion",
                ],
            )  # keep starting epoch for tensorboard log

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch

            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError("You need to specify the path to the first stage model.")

    extra_discriminators = {
        key: model[key]
        for key in ["msstft", "subband"]
        if key in model
    }
    gl = GeneratorLoss(model.mpd, model.msd, extra_discriminators).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd, extra_discriminators).to(device)
    wl = WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(
            sigma_min=0.0001, sigma_max=3.0, rho=9.0
        ),  # empirical parameters
        clamp=False,
    )

    grad_clip = float(getattr(optimizer_params, "grad_clip", 5.0))
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(getattr(optimizer_params, "pct_start", 0.0)),
        "final_div_factor": float(getattr(optimizer_params, "final_div_factor", 10)),
        "epochs": epochs,
        "steps_per_epoch": max(1, len(train_dataloader) // grad_accum),
    }
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    for key in ["bert_encoder", "bert", "predictor", "diffusion"]:
        if key in scheduler_params_dict:
            scheduler_params_dict[key]["steps_per_epoch"] = max(
                1, scheduler_params["steps_per_epoch"] * 2
            )
    scheduler_params_dict["bert"]["max_lr"] = optimizer_params.bert_lr * 2
    scheduler_params_dict["decoder"]["max_lr"] = optimizer_params.ft_lr * 2
    scheduler_params_dict["style_encoder"]["max_lr"] = optimizer_params.ft_lr * 2

    optimizer = build_optimizer(
        {key: model[key].parameters() for key in model},
        scheduler_params_dict=scheduler_params_dict,
        lr=optimizer_params.lr,
    )

    # adjust BERT learning rate
    for g in optimizer.optimizers["bert"].param_groups:
        g["betas"] = (0.9, 0.99)
        g["lr"] = optimizer_params.bert_lr
        g["initial_lr"] = optimizer_params.bert_lr
        g["min_lr"] = 0
        g["weight_decay"] = 0.01

    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g["betas"] = (0.0, 0.99)
            g["lr"] = optimizer_params.ft_lr
            g["initial_lr"] = optimizer_params.ft_lr
            g["min_lr"] = 0
            g["weight_decay"] = 1e-4

    # load models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(
            model,
            optimizer,
            config["pretrained_model"],
            load_only_params=config.get("load_only_params", True),
        )
        # Ensure all modules are in train() mode after loading
        # (load_checkpoint sets them to eval(), which breaks spectral_norm)
        [model[key].train() for key in model]

        # Initialize predictor_encoder from trained style_encoder
        # (predictor_encoder is not trained in Stage 1)
        model.predictor_encoder = copy.deepcopy(model.style_encoder)

    model = compile_model_for_training(model, config, logger)

    # DP — must happen AFTER load_checkpoint so state dict keys match
    # (DataParallel adds 'module.' prefix which breaks strict=False loading)
    discriminator_keys = {"mpd", "msd", "msstft", "subband", "wd"}
    for key in model:
        if key not in discriminator_keys:
            model[key] = MyDataParallel(model[key])

    n_down = model.text_aligner.n_down

    best_loss = float("inf")  # best test loss
    iters = 0

    torch.cuda.empty_cache()

    stft_loss = MultiResolutionSTFTLoss().to(device)

    print("BERT", optimizer.optimizers["bert"])
    print("decoder", optimizer.optimizers["decoder"])

    start_ds = True
    running_std = []

    slmadv_params = Munch(config["slmadv_params"])
    slmadv = SLMAdversarialLoss(
        model,
        wl,
        sampler,
        slmadv_params.min_len,
        slmadv_params.max_len,
        batch_percentage=slmadv_params.batch_percentage,
        skip_update=slmadv_params.iter,
        sig=slmadv_params.sig,
        diffusion_enabled=(diff_epoch < epochs),
    )

    # ── Kokoro-faithful TensorBoard inference setup ───────────────────────────
    text_cleaner = TextCleaner()
    _test_tokens = prepare_test_tokens(text_cleaner)
    if _test_tokens:
        logger.info(
            f"TensorBoard inference: prepared {len(_test_tokens)} test sentences"
        )
    else:
        logger.warning(
            "TensorBoard inference: no test sentences available (misaki/espeak not found?)"
        )

    # Generate Stage 1 baseline before any training updates
    [model[key].eval() for key in model]
    logger.info("Extracting Stage 1 baseline voicepack for TensorBoard...")
    _baseline_voicepack, _baseline_acoustic_norm, _baseline_prosodic_norm = (
        extract_voicepack(
            model,
            root_path,
            device,
            n_samples=200,
        )
    )
    if _baseline_voicepack is not None:
        writer.add_scalar("baseline/acoustic_norm", _baseline_acoustic_norm, 0)
        writer.add_scalar("baseline/prosodic_norm", _baseline_prosodic_norm, 0)
        logger.info(
            f"  acoustic_norm={_baseline_acoustic_norm:.4f}  prosodic_norm={_baseline_prosodic_norm:.4f}"
        )
        _baseline_audio = run_kokoro_inference(
            model, _test_tokens, _baseline_voicepack, device, text_cleaner
        )
        for i, (text, audio) in enumerate(_baseline_audio):
            writer.add_audio(f"baseline/test_{i + 1:02d}", audio, 0, sample_rate=sr)
            logger.info(f"  baseline/test_{i + 1:02d}: {text[:60]}")

    [model[key].train() for key in model]
    # ─────────────────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        running_steps = 0

        _ = [model[key].eval() for key in model]

        _ = [model[key].train() for key in model]

        adv_weight = linear_warmup(
            epoch,
            joint_epoch,
            adv_warmup_epochs,
            lambda_gen_start,
            1.0,
        )
        slm_weight = linear_warmup(
            epoch,
            joint_epoch,
            adv_warmup_epochs,
            lambda_slm_start,
            1.0,
        )
        slmadv_weight = linear_warmup(
            epoch,
            joint_epoch,
            slmadv_warmup_epochs,
            lambda_slm_start,
            1.0,
        )
        real_audio_mix = linear_warmup(
            epoch,
            joint_epoch,
            real_audio_mix_warmup_epochs,
            0.0,
            1.0,
        )
        extra_disc_weights = {
            "msstft": (
                float(getattr(loss_params, "lambda_msstft_adv", 0.10)),
                float(getattr(loss_params, "lambda_msstft_fm", 0.50)),
            ),
            "subband": (
                float(getattr(loss_params, "lambda_subband_adv", 0.05)),
                float(getattr(loss_params, "lambda_subband_fm", 0.25)),
            ),
        }

        train_bar = make_progress_bar(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Train {epoch + 1}/{epochs}",
        )
        for i, batch in train_bar:
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            (
                texts,
                input_lengths,
                ref_texts,
                ref_lengths,
                mels,
                mel_input_length,
                ref_mels,
            ) = batch

            # grad-accum window bookkeeping: zero grads at the window start before
            # any `continue`, so a skipped micro-batch can never leave already-stepped
            # grads to be re-applied (contamination) on the next window
            is_update_step = (i + 1) % grad_accum == 0 or (i + 1) == len(train_dataloader)
            # actual micro-batch count for this window (the last window of the epoch
            # may be shorter than grad_accum) so the loss average is correctly weighted
            accum_steps = min(grad_accum, len(train_dataloader) - (i // grad_accum) * grad_accum)
            if i % grad_accum == 0:
                if start_ds:
                    optimizer.optimizers["msd"].zero_grad()
                    optimizer.optimizers["mpd"].zero_grad()
                    for disc_key in extra_discriminators:
                        optimizer.optimizers[disc_key].zero_grad()
                for _k in ["bert_encoder", "bert", "predictor", "predictor_encoder", "diffusion", "style_encoder", "decoder"]:
                    if _k in optimizer.optimizers:
                        optimizer.optimizers[_k].zero_grad()

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2**n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

                try:
                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)
                except Exception:
                    continue

                mask_ST = mask_from_lens(
                    s2s_attn, input_lengths, mel_input_length // (2**n_down)
                )
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                asr = t_en @ s2s_attn_mono

                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                # compute reference styles
                ref: torch.Tensor | None = None
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref = torch.cat([ref_ss, ref_sp], dim=1)

            # compute the style of the entire utterance
            # this operation cannot be done in batch because of the avgpool layer (may need to work on masked avgpool)
            ss = []
            gs = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())
                mel = mels[bib, :, : mel_input_length[bib]]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)
                s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                gs.append(s)

            s_dur = torch.stack(ss).squeeze(1)  # global prosodic styles
            gs = torch.stack(gs).squeeze(1)  # global acoustic styles
            s_trg = torch.cat([gs, s_dur], dim=-1).detach()  # ground truth for denoiser

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            # denoiser training
            if epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)

                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.module.diffusion.sigma_data = (
                        s_trg.std(axis=-1).mean().item()
                    )  # batch-wise std estimation
                    running_std.append(model.diffusion.module.diffusion.sigma_data)

                if multispeaker:
                    s_preds = sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                        embedding=bert_dur,
                        embedding_scale=1,
                        features=ref,  # reference from the same speaker as the embedding
                        embedding_mask_proba=0.1,
                        num_steps=num_steps,
                    ).squeeze(1)
                    loss_diff = model.diffusion(
                        s_trg.unsqueeze(1), embedding=bert_dur, features=ref
                    ).mean()  # EDM loss
                    loss_sty = F.l1_loss(
                        s_preds, s_trg.detach()
                    )  # style reconstruction loss
                else:
                    s_preds = sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                        embedding=bert_dur,
                        embedding_scale=1,
                        embedding_mask_proba=0.1,
                        num_steps=num_steps,
                    ).squeeze(1)
                    loss_diff = model.diffusion.module.diffusion(
                        s_trg.unsqueeze(1), embedding=bert_dur
                    ).mean()  # EDM loss
                    loss_sty = F.l1_loss(
                        s_preds, s_trg.detach()
                    )  # style reconstruction loss
            else:
                loss_sty = 0
                loss_diff = 0

            d, p = model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)

            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)
            en, gt, st, p_en, wav = [], [], [], [], []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start : random_start + mel_len])
                p_en.append(p[bib, :, random_start : random_start + mel_len])
                gt.append(
                    mels[bib, :, (random_start * 2) : ((random_start + mel_len) * 2)]
                )

                y = waves[bib][
                    (random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300
                ]
                wav.append(torch.from_numpy(y).to(device))

                # style reference (better to be different from the GT)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(
                    mels[bib, :, (random_start * 2) : ((random_start + mel_len_st) * 2)]
                )

            wav = torch.stack(wav).float().detach()

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()

            if gt.size(-1) < 80:
                continue

            F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
            N_real = log_norm(gt.unsqueeze(1)).squeeze(1)

            s_dur = model.predictor_encoder(
                st.unsqueeze(1) if multispeaker else gt.unsqueeze(1)
            )
            s = model.style_encoder(
                st.unsqueeze(1) if multispeaker else gt.unsqueeze(1)
            )

            y_rec_gt = wav.unsqueeze(1)
            y_rec_gt_pred = model.decoder(en, F0_real, N_real, s)

            wav = real_audio_mix * y_rec_gt + (1.0 - real_audio_mix) * y_rec_gt_pred.detach()

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

            y_rec = model.decoder(en, F0_fake, N_fake, s)

            loss_F0_rec = (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            if start_ds:
                d_loss = adv_weight * (
                    dl(wav.detach(), y_rec.detach(), extra_disc_weights).mean()
                    / accum_steps
                )
                d_loss.backward()
                if is_update_step:
                    torch.nn.utils.clip_grad_norm_(
                        list(model["msd"].parameters())
                        + list(model["mpd"].parameters())
                        + [
                            p
                            for disc_key in extra_discriminators
                            for p in model[disc_key].parameters()
                        ],
                        grad_clip
                    )
                    optimizer.step_and_scheduler("msd")
                    optimizer.step_and_scheduler("mpd")
                    for disc_key in extra_discriminators:
                        optimizer.step_and_scheduler(disc_key)
            else:
                d_loss = 0

            loss_mel = stft_loss(y_rec, wav)
            if start_ds:
                loss_gen_all = gl(wav, y_rec, extra_disc_weights).mean()
            else:
                loss_gen_all = 0
            loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze()).mean()

            loss_ce = 0
            loss_dur = 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, : _text_input[p]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(
                    _dur_pred[1 : _text_length - 1], _text_input[1 : _text_length - 1]
                )
                loss_ce += F.binary_cross_entropy_with_logits(
                    _s2s_pred.flatten(), _s2s_trg.flatten()
                )

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)

            g_loss = (
                loss_params.lambda_mel * loss_mel
                + loss_params.lambda_F0 * loss_F0_rec
                + loss_params.lambda_ce * loss_ce
                + loss_params.lambda_norm * loss_norm_rec
                + loss_params.lambda_dur * loss_dur
                + loss_params.lambda_gen * adv_weight * loss_gen_all
                + loss_params.lambda_slm * slm_weight * loss_lm
                + loss_params.lambda_sty * loss_sty
                + loss_params.lambda_diff * loss_diff
            )

            running_loss += loss_mel.item()
            running_steps += 1
            (g_loss / accum_steps).backward()

            if is_update_step:
                gen_keys = ["bert_encoder", "bert", "predictor", "predictor_encoder", "style_encoder", "decoder"]
                if epoch >= diff_epoch:
                    gen_keys.append("diffusion")
                torch.nn.utils.clip_grad_norm_(
                    [p for k in gen_keys if k in model for p in model[k].parameters()], grad_clip
                )
                optimizer.step_and_scheduler("bert_encoder")
                optimizer.step_and_scheduler("bert")
                optimizer.step_and_scheduler("predictor")
                optimizer.step_and_scheduler("predictor_encoder")
                optimizer.step_and_scheduler("style_encoder")
                optimizer.step_and_scheduler("decoder")

                if epoch >= diff_epoch:
                    optimizer.step_and_scheduler("diffusion")

            if epoch >= joint_epoch:
                # randomly pick whether to use in-distribution text
                if np.random.rand() < 0.5:
                    use_ind = True
                else:
                    use_ind = False

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts

                if is_update_step:
                    slm_out = slmadv(
                        i // grad_accum,
                        y_rec_gt,
                        y_rec_gt_pred,
                        waves,
                        mel_input_length,
                        ref_texts,
                        ref_lengths,
                        use_ind,
                        s_trg.detach(),
                        ref if multispeaker else None,
                    )
                else:
                    slm_out = None

                if slm_out is None:
                    iters = iters + 1
                    reason = "no-update-step" if not is_update_step else "no-valid-clips"
                    train_bar.set_postfix(
                        metric_postfix(
                            mel=running_loss / max(1, running_steps),
                            disc=d_loss,
                            dur=loss_dur,
                        )
                        | {"slmadv": reason}
                    )
                    if (i + 1) % log_interval == 0:
                        running_loss = 0.0
                        running_steps = 0
                    continue

                d_loss_slm, loss_gen_lm, y_pred = slm_out
                d_loss_slm = d_loss_slm * slmadv_weight if d_loss_slm != 0 else d_loss_slm
                loss_gen_lm = loss_gen_lm * slmadv_weight

                # SLM generator loss
                for _k in ["bert_encoder", "bert", "predictor", "diffusion"]:
                    if _k in optimizer.optimizers:
                        optimizer.optimizers[_k].zero_grad()
                loss_gen_lm.backward()

                # compute the gradient norm
                total_norm = {}
                for key in model.keys():
                    total_norm[key] = 0
                    parameters = [
                        p
                        for p in model[key].parameters()
                        if p.grad is not None and p.requires_grad
                    ]
                    for p in parameters:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm[key] += param_norm.item() ** 2
                    total_norm[key] = total_norm[key] ** 0.5

                # gradient scaling
                if total_norm["predictor"] > slmadv_params.thresh:
                    for key in model.keys():
                        for p in model[key].parameters():
                            if p.grad is not None:
                                p.grad *= 1 / total_norm["predictor"]

                for p in model.predictor.duration_proj.parameters():
                    if p.grad is not None:
                        p.grad *= slmadv_params.scale

                for p in model.predictor.lstm.parameters():
                    if p.grad is not None:
                        p.grad *= slmadv_params.scale

                for p in model.diffusion.parameters():
                    if p.grad is not None:
                        p.grad *= slmadv_params.scale

                optimizer.step_and_scheduler("bert_encoder")
                optimizer.step_and_scheduler("bert")
                optimizer.step_and_scheduler("predictor")
                optimizer.step_and_scheduler("diffusion")

                # SLM discriminator loss
                if d_loss_slm != 0:
                    optimizer.optimizers["wd"].zero_grad()
                    d_loss_slm.backward(retain_graph=True)
                    optimizer.step_and_scheduler("wd")

            else:
                d_loss_slm, loss_gen_lm = 0, 0

            iters = iters + 1

            avg_mel_loss = running_loss / max(1, running_steps)
            train_bar.set_postfix(
                metric_postfix(
                    mel=avg_mel_loss,
                    disc=d_loss,
                    dur=loss_dur,
                    ce=loss_ce,
                    f0=loss_F0_rec,
                    lm=loss_lm,
                    gen=loss_gen_all,
                    sty=loss_sty,
                    diff=loss_diff,
                    slm_d=d_loss_slm,
                    slm_g=loss_gen_lm,
                )
            )

            if (i + 1) % log_interval == 0:

                writer.add_scalar("train/mel_loss", avg_mel_loss, iters)
                writer.add_scalar("train/gen_loss", metric_value(loss_gen_all), iters)
                writer.add_scalar("train/d_loss", metric_value(d_loss), iters)
                writer.add_scalar("train/ce_loss", metric_value(loss_ce), iters)
                writer.add_scalar("train/dur_loss", metric_value(loss_dur), iters)
                writer.add_scalar("train/slm_loss", metric_value(loss_lm), iters)
                writer.add_scalar("train/norm_loss", metric_value(loss_norm_rec), iters)
                writer.add_scalar("train/F0_loss", metric_value(loss_F0_rec), iters)
                writer.add_scalar("train/sty_loss", metric_value(loss_sty), iters)
                writer.add_scalar("train/diff_loss", metric_value(loss_diff), iters)
                writer.add_scalar("train/d_loss_slm", metric_value(d_loss_slm), iters)
                writer.add_scalar("train/gen_loss_slm", metric_value(loss_gen_lm), iters)
                writer.add_scalar("train/adv_weight", adv_weight, iters)
                writer.add_scalar("train/slm_weight", slm_weight, iters)
                writer.add_scalar("train/slmadv_weight", slmadv_weight, iters)
                writer.add_scalar("train/real_audio_mix", real_audio_mix, iters)
                for lr_key in ["decoder", "msd", "mpd", "wd"]:
                    if lr_key in optimizer.optimizers:
                        writer.add_scalar(
                            f"lr/{lr_key}",
                            optimizer.optimizers[lr_key].param_groups[0]["lr"],
                            iters,
                        )

                running_loss = 0.0
                running_steps = 0

        loss_test = 0
        loss_align = 0
        loss_f = 0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            val_bar = make_progress_bar(
                enumerate(val_dataloader),
                total=len(val_dataloader),
                desc=f"Eval {epoch + 1}/{epochs}",
            )
            for batch_idx, batch in val_bar:
                optimizer.zero_grad()

                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    (
                        texts,
                        input_lengths,
                        ref_texts,
                        ref_lengths,
                        mels,
                        mel_input_length,
                        ref_mels,
                    ) = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2**n_down)).to(
                            "cuda"
                        )
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_ST = mask_from_lens(
                            s2s_attn, input_lengths, mel_input_length // (2**n_down)
                        )
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = t_en @ s2s_attn_mono

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss = []
                    gs = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item())
                        mel = mels[bib, :, : mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    s = torch.stack(ss).squeeze(1)
                    gs = torch.stack(gs).squeeze(1)
                    s_trg = torch.cat([s, gs], dim=-1).detach()

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
                    d, p = model.predictor(
                        d_en, s, input_lengths, s2s_attn_mono, text_mask
                    )
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []
                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start : random_start + mel_len])
                        p_en.append(p[bib, :, random_start : random_start + mel_len])

                        gt.append(
                            mels[
                                bib,
                                :,
                                (random_start * 2) : ((random_start + mel_len) * 2),
                            ]
                        )

                        y = waves[bib][
                            (random_start * 2) * 300 : ((random_start + mel_len) * 2)
                            * 300
                        ]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()

                    s = model.predictor_encoder(gt.unsqueeze(1))

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(
                        d, (d_gt), input_lengths
                    ):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, : _text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(
                            _dur_pred[1 : _text_length - 1],
                            _text_input[1 : _text_length - 1],
                        )

                    loss_dur /= texts.size(0)

                    s = model.style_encoder(gt.unsqueeze(1))

                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1
                    val_bar.set_postfix(
                        metric_postfix(
                            mel=loss_test / max(1, iters_test),
                            dur=loss_align / max(1, iters_test),
                            f0=loss_f / max(1, iters_test),
                        )
                    )
                except Exception as e:
                    tqdm_msg = f"validation batch skipped: {e}"
                    val_bar.write(tqdm_msg)
                    logger.debug(tqdm_msg)
                    logger.debug(traceback.format_exc())
                    continue

        print("Epochs:", epoch + 1)
        logger.info(
            "Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f"
            % (loss_test / iters_test, loss_align / iters_test, loss_f / iters_test)
            + "\n\n\n"
        )
        print("\n\n\n")
        writer.add_scalar("eval/mel_loss", loss_test / iters_test, epoch + 1)
        writer.add_scalar("eval/dur_loss", loss_align / iters_test, epoch + 1)
        writer.add_scalar("eval/F0_loss", loss_f / iters_test, epoch + 1)

        if epoch % saving_epoch == 0:
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print("Saving..")
            state = {
                "net": {key: model[key].state_dict() for key in model},
                "optimizer": optimizer.state_dict(),
                "iters": iters,
                "val_loss": loss_test / iters_test,
                "epoch": epoch,
            }
            save_path = osp.join(log_dir, "epoch_2nd_%05d.pth" % epoch)
            torch.save(state, save_path)

            # if estimate sigma, save the estimated sigma
            if model_params.diffusion.dist.estimate_sigma_data:
                config["model_params"]["diffusion"]["dist"]["sigma_data"] = float(
                    np.mean(running_std)
                )

                with open(osp.join(log_dir, osp.basename(config_path)), "w") as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

        # ── Kokoro-faithful TensorBoard inference (every epoch) ───────────────
        [model[key].eval() for key in model]
        logger.info(f"Epoch {epoch}: extracting voicepack for TensorBoard inference...")
        _vp, _acoustic_norm, _prosodic_norm = extract_voicepack(
            model,
            root_path,
            device,
            n_samples=200,
        )
        writer.add_scalar("voicepack/acoustic_norm", _acoustic_norm, epoch + 1)
        writer.add_scalar("voicepack/prosodic_norm", _prosodic_norm, epoch + 1)
        logger.info(
            f"  acoustic_norm={_acoustic_norm:.4f}  prosodic_norm={_prosodic_norm:.4f}"
        )
        if _vp is not None and _test_tokens:
            _epoch_audio = run_kokoro_inference(
                model, _test_tokens, _vp, device, text_cleaner
            )
            for i, (text, audio) in enumerate(_epoch_audio):
                writer.add_audio(
                    f"inference/test_{i + 1:02d}", audio, epoch + 1, sample_rate=sr
                )

        [model[key].train() for key in model]
        # ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
