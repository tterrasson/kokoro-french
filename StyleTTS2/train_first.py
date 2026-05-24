import logging
import os
import os.path as osp
import random
import shutil
import time
import warnings

import click
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from kokoro_symbols import TextCleaner
from kokoro_tb_utils import extract_voicepack, prepare_test_tokens, run_kokoro_inference
from losses import DiscriminatorLoss, GeneratorLoss, MultiResolutionSTFTLoss, WavLMLoss
from meldataset import build_dataloader
from models import build_model, load_ASR_models, load_checkpoint, load_F0_models
from munch import Munch
from optimizers import build_optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from utils import (
    get_data_path_list,
    get_image,
    length_to_mask,
    log_norm,
    log_print,
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

logger = get_logger(__name__, log_level="DEBUG")


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

    grad_accum = config.get("grad_accum", 1)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_dir=log_dir,
        split_batches=True,
        gradient_accumulation_steps=grad_accum,
        kwargs_handlers=[ddp_kwargs],
    )

    if run_name is None:
        run_name = time.strftime("%Y%m%d_%H%M%S")

    writer: SummaryWriter | None = None
    if accelerator.is_main_process:
        writer = SummaryWriter(osp.join(log_dir, "tensorboard", run_name))

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, "train.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(levelname)s:%(asctime)s: %(message)s")
    )
    logger.logger.addHandler(file_handler)

    batch_size = config.get("batch_size", 10)
    device = accelerator.device

    epochs = config.get("epochs_1st", 200)
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

    max_len = config.get("max_len", 200)

    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)

    train_dataloader = build_dataloader(
        train_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_config={},
        device=device,
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
    )

    with accelerator.main_process_first():
        # load pretrained ASR model
        ASR_config = config.get("ASR_config", False)
        ASR_path = config.get("ASR_path", False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = config.get("F0_path", False)
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        BERT_path = config.get("PLBERT_dir", False)
        plbert = load_plbert(BERT_path)

    grad_clip = float(config["optimizer_params"].get("grad_clip", 5.0))
    scheduler_params = {
        "max_lr": float(config["optimizer_params"].get("lr", 1e-4)),
        "pct_start": float(config["optimizer_params"].get("pct_start", 0.0)),
        "final_div_factor": float(config["optimizer_params"].get("final_div_factor", 10)),
        "epochs": epochs,
        "steps_per_epoch": max(1, len(train_dataloader) // grad_accum),
    }

    model_params = recursive_munch(config["model_params"])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    best_loss = float("inf")  # best test loss

    loss_params = Munch(config["loss_params"])
    TMA_epoch = loss_params.TMA_epoch
    adv_warmup_epochs = int(getattr(loss_params, "adv_warmup_epochs", 3))
    lambda_gen_start = float(getattr(loss_params, "lambda_gen_start", 0.05))
    lambda_slm_start = float(getattr(loss_params, "lambda_slm_start", 0.05))

    for k in model:
        model[k] = accelerator.prepare(model[k])

    train_dataloader, val_dataloader = accelerator.prepare(
        train_dataloader, val_dataloader
    )

    [model[key].to(device) for key in model]

    # initialize optimizers after preparing models for compatibility with FSDP
    optimizer = build_optimizer(
        {key: model[key].parameters() for key in model},
        scheduler_params_dict={key: scheduler_params.copy() for key in model},
        lr=float(config["optimizer_params"].get("lr", 1e-4)),
    )

    for k, v in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    with accelerator.main_process_first():
        if config.get("pretrained_model", "") != "":
            model, optimizer, start_epoch, iters = load_checkpoint(
                model,
                optimizer,
                config["pretrained_model"],
                load_only_params=config.get("load_only_params", True),
            )
        else:
            start_epoch = 0
            iters = 0

    # in case not distributed
    try:
        n_down = model.text_aligner.module.n_down
    except AttributeError:
        n_down = model.text_aligner.n_down

    # wrapped losses for compatibility with mixed precision
    stft_loss = MultiResolutionSTFTLoss().to(device)
    extra_discriminators = {
        key: model[key]
        for key in ["msstft", "subband"]
        if key in model
    }
    gl = GeneratorLoss(model.mpd, model.msd, extra_discriminators).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd, extra_discriminators).to(device)
    wl = WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr).to(device)

    # ── Kokoro-faithful TensorBoard inference setup ───────────────────────────
    text_cleaner: TextCleaner | None = None
    _test_tokens: list = []
    if accelerator.is_main_process:
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

        # Generate pretrained baseline before any training updates
        _ = [model[key].eval() for key in model]
        logger.info("Extracting pretrained baseline voicepack for TensorBoard...")
        _baseline_voicepack, _baseline_acoustic_norm, _baseline_prosodic_norm = (
            extract_voicepack(model, root_path, device, n_samples=200)
        )
        if _baseline_voicepack is not None:
            assert writer is not None
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
        _ = [model[key].train() for key in model]
    # ─────────────────────────────────────────────────────────────────────────

    epoch = start_epoch
    loss_test = 0.0
    iters_test = 1
    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()
        adv_weight = linear_warmup(
            epoch,
            TMA_epoch,
            adv_warmup_epochs,
            lambda_gen_start,
            1.0,
        )
        slm_weight = linear_warmup(
            epoch,
            TMA_epoch,
            adv_warmup_epochs,
            lambda_slm_start,
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

        _ = [model[key].train() for key in model]

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, _, _, mels, mel_input_length, _ = batch

            # grad-accum window bookkeeping: zero grads at the window start before
            # any `continue`, so a skipped micro-batch can never leave already-stepped
            # grads to be re-applied (contamination) on the next window
            is_update_step = (i + 1) % grad_accum == 0 or (i + 1) == len(train_dataloader)
            # actual micro-batch count for this window (the last window of the epoch
            # may be shorter than grad_accum) so the loss average is correctly weighted
            accum_steps = min(grad_accum, len(train_dataloader) - (i // grad_accum) * grad_accum)
            if i % grad_accum == 0:
                optimizer.optimizers["msd"].zero_grad()
                optimizer.optimizers["mpd"].zero_grad()
                for disc_key in extra_discriminators:
                    optimizer.optimizers[disc_key].zero_grad()
                for _k in ["text_encoder", "style_encoder", "decoder", "text_aligner", "pitch_extractor"]:
                    optimizer.optimizers[_k].zero_grad()

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2**n_down)).to("cuda")
                text_mask = length_to_mask(input_lengths).to(texts.device)

            ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)

            with torch.no_grad():
                attn_mask = (
                    (~mask)
                    .unsqueeze(-1)
                    .expand(mask.shape[0], mask.shape[1], text_mask.shape[-1])
                    .float()
                    .transpose(-1, -2)
                )
                attn_mask = (
                    attn_mask.float()
                    * (~text_mask)
                    .unsqueeze(-1)
                    .expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1])
                    .float()
                )
                attn_mask = attn_mask < 1

            s2s_attn.masked_fill_(attn_mask, 0.0)

            with torch.no_grad():
                mask_ST = mask_from_lens(
                    s2s_attn, input_lengths, mel_input_length // (2**n_down)
                )
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)

            # 50% of chance of using monotonic version
            if bool(random.getrandbits(1)):
                asr = t_en @ s2s_attn
            else:
                asr = t_en @ s2s_attn_mono

            # get clips
            mel_input_length_all = accelerator.gather(
                mel_input_length
            )  # for balanced load
            mel_len = min(
                [int(mel_input_length_all.min().item() / 2 - 1), max_len // 2]
            )
            mel_len_st = int(mel_input_length.min().item() / 2 - 1)

            en = []
            gt = []
            wav = []
            st = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start : random_start + mel_len])
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

            en = torch.stack(en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()

            wav = torch.stack(wav).float().detach()

            # clip too short to be used by the style encoder
            if gt.shape[-1] < 80:
                continue

            with torch.no_grad():
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))

            s = model.style_encoder(
                st.unsqueeze(1) if multispeaker else gt.unsqueeze(1)
            )

            y_rec = model.decoder(en, F0_real, real_norm, s)

            # discriminator loss — accumulate over grad_accum steps, step only at boundary
            if epoch >= TMA_epoch:
                d_loss = adv_weight * (
                    dl(
                        wav.detach().unsqueeze(1).float(),
                        y_rec.detach(),
                        extra_disc_weights,
                    ).mean()
                    / accum_steps
                )
                accelerator.backward(d_loss)
                if is_update_step:
                    accelerator.clip_grad_norm_(
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

            loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

            if epoch >= TMA_epoch:  # start TMA training
                loss_s2s = 0
                for _s2s_pred, _text_input, _text_length in zip(
                    s2s_pred, texts, input_lengths
                ):
                    loss_s2s += F.cross_entropy(
                        _s2s_pred[:_text_length], _text_input[:_text_length]
                    )
                loss_s2s /= texts.size(0)

                loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

                loss_gen_all = gl(
                    wav.detach().unsqueeze(1).float(),
                    y_rec,
                    extra_disc_weights,
                ).mean()
                loss_slm = wl(wav.detach(), y_rec).mean()

                g_loss = (
                    loss_params.lambda_mel * loss_mel
                    + loss_params.lambda_mono * loss_mono
                    + loss_params.lambda_s2s * loss_s2s
                    + loss_params.lambda_gen * adv_weight * loss_gen_all
                    + loss_params.lambda_slm * slm_weight * loss_slm
                )

            else:
                loss_s2s = 0
                loss_mono = 0
                loss_gen_all = 0
                loss_slm = 0
                g_loss = loss_mel

            running_loss += accelerator.gather(loss_mel).mean().item()

            accelerator.backward(g_loss / accum_steps)

            if is_update_step:
                gen_keys = ["text_encoder", "style_encoder", "decoder"]
                if epoch >= TMA_epoch:
                    gen_keys += ["text_aligner", "pitch_extractor"]
                accelerator.clip_grad_norm_(
                    [p for k in gen_keys for p in model[k].parameters()], grad_clip
                )
                optimizer.step_and_scheduler("text_encoder")
                optimizer.step_and_scheduler("style_encoder")
                optimizer.step_and_scheduler("decoder")

                if epoch >= TMA_epoch:
                    optimizer.step_and_scheduler("text_aligner")
                    optimizer.step_and_scheduler("pitch_extractor")

            iters = iters + 1

            if (i + 1) % log_interval == 0 and accelerator.is_main_process:
                log_print(
                    "Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f"
                    % (
                        epoch + 1,
                        epochs,
                        i + 1,
                        len(train_list) // batch_size,
                        running_loss / log_interval,
                        loss_gen_all,
                        d_loss,
                        loss_mono,
                        loss_s2s,
                        loss_slm,
                    ),
                    logger,
                )

                assert writer is not None
                writer.add_scalar("train/mel_loss", running_loss / log_interval, iters)
                writer.add_scalar("train/gen_loss", loss_gen_all, iters)
                writer.add_scalar("train/d_loss", d_loss, iters)
                writer.add_scalar("train/mono_loss", loss_mono, iters)
                writer.add_scalar("train/s2s_loss", loss_s2s, iters)
                writer.add_scalar("train/slm_loss", loss_slm, iters)
                writer.add_scalar("train/adv_weight", adv_weight, iters)
                writer.add_scalar("train/slm_weight", slm_weight, iters)
                for lr_key in ["decoder", "msd", "mpd"]:
                    if lr_key in optimizer.optimizers:
                        writer.add_scalar(
                            f"lr/{lr_key}",
                            optimizer.optimizers[lr_key].param_groups[0]["lr"],
                            iters,
                        )

                running_loss = 0

                print("Time elasped:", time.time() - start_time)

        loss_test = 0

        _ = [model[key].eval() for key in model]

        s2s_attn = None
        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                texts, input_lengths, _, _, mels, mel_input_length, _ = batch

                with torch.no_grad():
                    mask = length_to_mask(mel_input_length // (2**n_down)).to("cuda")
                    ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)

                    text_mask = length_to_mask(input_lengths).to(texts.device)
                    attn_mask = (
                        (~mask)
                        .unsqueeze(-1)
                        .expand(mask.shape[0], mask.shape[1], text_mask.shape[-1])
                        .float()
                        .transpose(-1, -2)
                    )
                    attn_mask = (
                        attn_mask.float()
                        * (~text_mask)
                        .unsqueeze(-1)
                        .expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1])
                        .float()
                    )
                    attn_mask = attn_mask < 1
                    s2s_attn.masked_fill_(attn_mask, 0.0)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)

                asr = t_en @ s2s_attn

                # get clips
                mel_input_length_all = accelerator.gather(
                    mel_input_length
                )  # for balanced load
                mel_len = min(
                    [int(mel_input_length.min().item() / 2 - 1), max_len // 2]
                )

                en = []
                gt = []
                wav = []
                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item() / 2)

                    random_start = np.random.randint(0, mel_length - mel_len)
                    en.append(asr[bib, :, random_start : random_start + mel_len])
                    gt.append(
                        mels[
                            bib, :, (random_start * 2) : ((random_start + mel_len) * 2)
                        ]
                    )
                    y = waves[bib][
                        (random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300
                    ]
                    wav.append(torch.from_numpy(y).to("cuda"))

                wav = torch.stack(wav).float().detach()

                en = torch.stack(en)
                gt = torch.stack(gt).detach()

                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                s = model.style_encoder(gt.unsqueeze(1))
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                y_rec = model.decoder(en, F0_real, real_norm, s)

                loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                loss_test += accelerator.gather(loss_mel).mean().item()
                iters_test += 1

        if accelerator.is_main_process:
            print("Epochs:", epoch + 1)
            log_print(
                "Validation loss: %.3f" % (loss_test / iters_test), logger
            )
            print("\n\n")
            assert writer is not None
            assert s2s_attn is not None

            writer.add_scalar("eval/mel_loss", loss_test / iters_test, epoch + 1)
            attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
            writer.add_figure("eval/attn", attn_image, epoch)

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
                save_path = osp.join(log_dir, "epoch_1st_%05d.pth" % epoch)
                torch.save(state, save_path)

            # ── Kokoro-faithful TensorBoard inference (every epoch) ───────────
            _ = [model[key].eval() for key in model]
            logger.info(
                f"Epoch {epoch}: extracting voicepack for TensorBoard inference..."
            )
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
                        f"inference/test_{i + 1:02d}",
                        audio,
                        epoch + 1,
                        sample_rate=sr,
                    )
            _ = [model[key].train() for key in model]
            # ───────────────────────────────────────────────────────────────────

    if accelerator.is_main_process:
        print("Saving..")
        state = {
            "net": {key: model[key].state_dict() for key in model},
            "optimizer": optimizer.state_dict(),
            "iters": iters,
            "val_loss": loss_test / iters_test,
            "epoch": epoch,
        }
        save_path = osp.join(log_dir, config.get("first_stage_path", "first_stage.pth"))
        torch.save(state, save_path)


if __name__ == "__main__":
    main()
