import torch
import numpy as np
from tqdm import tqdm
import time
import random
import pandas as pd
from collections import defaultdict
from mel_dataset import MelDataset, collect_audio_files, MelSpectrogramConverter
from models import (
    generator_v1,
    generator_v2,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)

from torch.nn import L1Loss

import logging
import sys
import os

DATASET = "/home/SSD/new_location/"

SEGMENT_SIZE = 8192
SAMPLING_RATE = 22050
N_FFT = 1024
WIN_SIZE = 1024
NUM_MELS = 80
FMIN = 0
FMAX = 8000
HOP_SIZE = 256

DEVICE = "cuda:2"
SEED = 3
EPOCHS = 100
LR_RATE = 0.0002
BETAS = [0.8, 0.99]
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAIN_RATIO = 0.99
DECAY = 0.999
L1_WEIGHT = 45
FINETUNE = True

DETERMINISTIC = False

GENERATOR = generator_v2
GENERATOR_PT_FILE = "hifigan_lj_v2.pt"  # "g_02500000"
LOAD_OPTIMIZER = False
DISCRIMINATOR_PT_FILE = "do_02500000"


def setup_logger(log_file="training.log"):
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Set up the logger
logger = setup_logger()


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(DETERMINISTIC)
    if not DETERMINISTIC:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_nan(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            return True
    return False


def train_one_epoch(
    generator,
    mpd,
    msd,
    train_loader,
    optim_d,
    optim_g,
    scaler,
    scheduler_g,
    scheduler_d,
):
    generator.train()
    mpd.train()
    msd.train()

    l1_loss = L1Loss()
    mel_converter_loss = MelSpectrogramConverter(
        N_FFT,
        NUM_MELS,
        SAMPLING_RATE,
        HOP_SIZE,
        WIN_SIZE,
        FMIN,
        None,
    )
    total_losses = defaultdict(float)

    for audio in tqdm(train_loader, desc="Training"):
        audio = [k.to(DEVICE) for k in audio]
        (mel, audio, mel_for_loss) = audio

        optim_d.zero_grad()

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            y_g_hat = generator(mel)
            y_g_hat_mel = mel_converter_loss(y_g_hat)

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(
                audio[:, None], y_g_hat.detach()[:, None]
            )
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g
            )

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(
                audio[:, None], y_g_hat.detach()[:, None]
            )
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g
            )

            loss_disc_all = loss_disc_s + loss_disc_f

        scaler.scale(loss_disc_all).backward()
        scaler.step(optim_d)

        if check_nan(mpd) or check_nan(msd):
            logger.error("NaN detected in discriminator parameters. Skipping batch.")
            continue

        optim_g.zero_grad()

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            # L1 Mel-Spectrogram Loss
            loss_mel = l1_loss(mel_for_loss, y_g_hat_mel) * L1_WEIGHT

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
                audio[:, None], y_g_hat[:, None]
            )
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(
                audio[:, None], y_g_hat[:, None]
            )
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        scaler.scale(loss_gen_all).backward()
        scaler.step(optim_g)

        if check_nan(generator):
            logger.error("NaN detected in generator parameters. Skipping batch.")
            continue

        scaler.update()

        batch_size = audio.size(0)
        for loss_name, loss_value in [
            # Generator losses
            ("train_loss_gen_s", loss_gen_s),
            ("train_loss_gen_f", loss_gen_f),
            ("train_loss_fm_s", loss_fm_s),
            ("train_loss_fm_f", loss_fm_f),
            ("train_loss_mel", loss_mel),
            ("train_loss_gen_all", loss_gen_all),
            # Discriminator losses
            ("train_loss_disc_s", loss_disc_s),
            ("train_loss_disc_f", loss_disc_f),
            ("train_loss_disc_all", loss_disc_all),
        ]:
            total_losses[loss_name] += loss_value.item() * batch_size

    scheduler_g.step()
    scheduler_d.step()

    total_samples = len(train_loader.dataset)
    return {k: v / total_samples for k, v in total_losses.items()}


def val_one_epoch(generator, val_loader):
    generator.eval()
    l1_loss = L1Loss()
    total_losses = defaultdict(float)
    mel_converter_loss = MelSpectrogramConverter(
        N_FFT,
        NUM_MELS,
        SAMPLING_RATE,
        HOP_SIZE,
        WIN_SIZE,
        FMIN,
        None,
    )

    with torch.inference_mode(), torch.cuda.amp.autocast(
        enabled=True, dtype=torch.float16
    ):
        for audio in tqdm(val_loader, desc="Validation"):
            audio = [k.to(DEVICE) for k in audio]
            (mel, audio, mel_for_loss) = audio

            y_g_hat = generator(mel)
            y_g_hat_mel = mel_converter_loss(y_g_hat)
            val_err_tot = l1_loss(mel_for_loss, y_g_hat_mel)

            batch_size = audio.size(0)
            for loss_name, loss_value in [
                ("val_mel_loss", val_err_tot),
            ]:
                total_losses[loss_name] += loss_value.item() * batch_size

    total_samples = len(val_loader.dataset)
    losses = {k: v / total_samples for k, v in total_losses.items()}
    losses["val_total_loss"] = sum(losses.values())

    return losses


def main():
    start_time = time.time()

    generator = GENERATOR().to(DEVICE)
    mpd = MultiPeriodDiscriminator().to(DEVICE)
    msd = MultiScaleDiscriminator().to(DEVICE)
    logger.info(generator)

    optim_g = torch.optim.AdamW(generator.parameters(), LR_RATE, betas=BETAS)
    optim_d = torch.optim.AdamW(
        list(msd.parameters()) + list(mpd.parameters()),
        LR_RATE,
        betas=BETAS,
    )

    last_epoch = -1
    if os.path.exists(DISCRIMINATOR_PT_FILE):
        logger.info(f"Loading discriminator: {DISCRIMINATOR_PT_FILE}")
        state_dict_do = torch.load(DISCRIMINATOR_PT_FILE, map_location=DEVICE)
        mpd.load_state_dict(state_dict_do["mpd"])
        msd.load_state_dict(state_dict_do["msd"])
        if LOAD_OPTIMIZER:
            optim_d.load_state_dict(state_dict_do["optim_d"])
            optim_g.load_state_dict(state_dict_do["optim_g"])
            last_epoch = state_dict_do["epoch"]
    if os.path.exists(GENERATOR_PT_FILE):
        logger.info(f"Loading generator: {GENERATOR_PT_FILE}")
        state_dict_g = torch.load(GENERATOR_PT_FILE, map_location=DEVICE)
        generator.load_state_dict(state_dict_g["generator"])

    total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}")

    train_files, val_files = collect_audio_files(DATASET, train_ratio=TRAIN_RATIO)
    logger.info(
        f"Training files: {len(train_files)}, validation files: {len(val_files)}"
    )

    seed_all(SEED)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_dataset = MelDataset(
        train_files,
        SEGMENT_SIZE,
        N_FFT,
        NUM_MELS,
        HOP_SIZE,
        WIN_SIZE,
        SAMPLING_RATE,
        FMIN,
        FMAX,
        split=True,
        fmax_loss=None,
        fine_tuning=FINETUNE,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        collate_fn=train_dataset.collate_fn,
        generator=g,
    )

    val_dataset = MelDataset(
        val_files,
        SEGMENT_SIZE,
        N_FFT,
        NUM_MELS,
        HOP_SIZE,
        WIN_SIZE,
        SAMPLING_RATE,
        FMIN,
        FMAX,
        split=False,
        fmax_loss=None,
        fine_tuning=FINETUNE,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        collate_fn=val_dataset.collate_fn,
        generator=g,
    )

    scaler = torch.cuda.amp.GradScaler()

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=DECAY, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=DECAY, last_epoch=last_epoch
    )

    best_loss = float("inf")
    log_file = []
    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Epoch: {epoch}/{EPOCHS}")
        epoch_start_time = time.time()

        epoch_info = {"epoch": epoch}

        epoch_info |= train_one_epoch(
            generator,
            mpd,
            msd,
            train_loader,
            optim_d,
            optim_g,
            scaler,
            scheduler_g,
            scheduler_d,
        )
        epoch_info |= val_one_epoch(generator, val_loader)

        if epoch_info["val_total_loss"] < best_loss:
            best_loss = epoch_info["val_total_loss"]
            logger.info("New best val_total_loss")
            torch.save(generator.state_dict(), "model_generator.pt")
            torch.save(
                {
                    "mpd": mpd.state_dict(),
                    "msd": msd.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                }
                | epoch_info,
                "model_discriminator.pt",
            )

        log_file.append(
            epoch_info
            | {
                "elapsed": (time.time() - epoch_start_time) / 60,
                "elapsed_total": (time.time() - start_time) / 60,
                "lr_g": scheduler_g.get_last_lr()[0],
                "lr_d": scheduler_d.get_last_lr()[0],
            }
        )
        logger.info(log_file[-1])

        for k, v in epoch_info.items():
            if np.isnan(v):
                logger.error("Found NaN!")
                break

    pd.DataFrame(log_file).to_csv("model.csv", index=False)

    logger.info(f"Best loss: {best_loss}")
    run_time = (time.time() - start_time) / 60

    logger.info(f"Run time: {run_time} minutes")


if __name__ == "__main__":
    main()
