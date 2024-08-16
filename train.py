import os
import time
import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from meldataset import MelDataset, collect_audio_files, MelSpectrogramConverter
from models import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)
from tqdm import tqdm
import shutil
from pathlib import Path
from typing import Dict, Any


def load_checkpoint(filepath: str, device: torch.device) -> Dict[str, Any]:
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    print(f"Loading checkpoint: {filepath}")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Checkpoint loaded successfully.")
    return checkpoint_dict


def save_checkpoint(filepath: str, obj: Dict[str, Any]) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to: {filepath}")
    torch.save(obj, filepath)
    print("Checkpoint saved successfully.")


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def setup_training_environment(
    config_path: str, config_name: str, output_path: str
) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_path / config_name)


def setup_logger(log_dir: str) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    handler = RotatingFileHandler(log_file, maxBytes=2**20, backupCount=3)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def train(a, h):
    logger = setup_logger(a.checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        list(msd.parameters()) + list(mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    logger.info(str(generator))
    logger.info(f"Checkpoints directory: {a.checkpoint_path}")

    steps = 0
    last_epoch = -1
    if os.path.exists(a.pretrained_discriminator):
        logger.info(f"Loading discriminator: {a.pretrained_discriminator}")
        state_dict_do = load_checkpoint(a.pretrained_discriminator, device)
        mpd.load_state_dict(state_dict_do["mpd"])
        msd.load_state_dict(state_dict_do["msd"])
        optim_d.load_state_dict(state_dict_do["optim_d"])
        steps = state_dict_do["steps"]
        last_epoch = state_dict_do["epoch"]
    if os.path.exists(a.pretrained_generator):
        logger.info(f"Loading generator: {a.pretrained_generator}")
        state_dict_g = load_checkpoint(a.pretrained_generator, device)
        generator.load_state_dict(state_dict_g["generator"])
        optim_g.load_state_dict(state_dict_do["optim_g"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )

    logger.info(str(a))
    training_filelist, validation_filelist = collect_audio_files(a.input_wavs_dir)

    trainset = MelDataset(
        training_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        n_cache_reuse=0,
        shuffle=True,
        fmax_loss=h.fmax_for_loss,
        device=device,
        fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir,
    )

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=True,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    validset = MelDataset(
        validation_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        False,
        False,
        n_cache_reuse=0,
        fmax_loss=h.fmax_for_loss,
        device=device,
        fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir,
    )
    validation_loader = DataLoader(
        validset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
    )

    generator.train()
    mpd.train()
    msd.train()

    mel_converter_loss = MelSpectrogramConverter(
        h.n_fft,
        h.num_mels,
        h.sampling_rate,
        h.hop_size,
        h.win_size,
        h.fmin,
        h.fmax_for_loss,
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(max(0, last_epoch + 1), a.training_epochs):
        start = time.time()
        logger.info(f"Epoch: {epoch + 1}")

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for i, batch in enumerate(pbar):
            start_b = time.time()
            x, y, _, y_mel = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)

            with torch.cuda.amp.autocast():
                y_g_hat = generator(x)
                y_g_hat_mel = mel_converter_loss(y_g_hat.squeeze(1))

                optim_d.zero_grad()

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )

                loss_disc_all = loss_disc_s + loss_disc_f

            scaler.scale(loss_disc_all).backward()
            scaler.step(optim_d)

            with torch.cuda.amp.autocast():
                optim_g.zero_grad()

                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = (
                    loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                )

            scaler.scale(loss_gen_all).backward()
            scaler.step(optim_g)
            scaler.update()

            with torch.no_grad():
                mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

            # Update tqdm progress bar
            pbar.set_postfix(
                {
                    "Gen Loss": f"{loss_gen_all.item():.4f}",
                    "Disc Loss": f"{loss_disc_all.item():.4f}",
                    "Mel Error": f"{mel_error:.4f}",
                }
            )

            if steps % a.stdout_interval == 0:
                logger.info(
                    f"Steps : {steps}, Gen Loss Total : {loss_gen_all:.4f}, Disc Loss: {loss_disc_all:.4f}, "
                    f"Mel-Spec. Error : {mel_error:.4f}, s/b : {time.time() - start_b:.4f}"
                )

            if steps % a.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = f"{a.checkpoint_path}/g_{steps:08d}"
                save_checkpoint(
                    checkpoint_path,
                    {"generator": generator.state_dict()},
                )
                checkpoint_path = f"{a.checkpoint_path}/do_{steps:08d}"
                save_checkpoint(
                    checkpoint_path,
                    {
                        "mpd": mpd.state_dict(),
                        "msd": msd.state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "optim_d": optim_d.state_dict(),
                        "steps": steps,
                        "epoch": epoch,
                    },
                )

            if steps % a.validation_interval == 0 and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        for j, batch in enumerate(
                            tqdm(validation_loader, desc="Validation")
                        ):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))
                            y_mel = y_mel.to(device, non_blocking=True)
                            y_g_hat_mel = mel_converter_loss(y_g_hat.squeeze(1))
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                    val_err = val_err_tot / (j + 1)
                    logger.info(f"Validation Mel-Spectrogram Error: {val_err:.4f}")

                generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        logger.info(
            f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--group_name", default=None)
    parser.add_argument("--input_wavs_dir", default="LJSpeech-1.1/wavs")
    parser.add_argument("--input_mels_dir", default="ft_dataset")
    parser.add_argument("--input_training_file", default="LJSpeech-1.1/training.txt")
    parser.add_argument(
        "--input_validation_file", default="LJSpeech-1.1/validation.txt"
    )
    parser.add_argument("--checkpoint_path", default="cp_hifigan")
    parser.add_argument("--pretrained_discriminator", default="cp_hifigan/do_02500000")
    parser.add_argument("--pretrained_generator", default="cp_hifigan/g_02500000")
    parser.add_argument("--config", default="")
    parser.add_argument("--training_epochs", default=100, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=5000, type=int)
    parser.add_argument("--summary_interval", default=100, type=int)
    parser.add_argument("--validation_interval", default=5000, type=int)
    parser.add_argument("--fine_tuning", default=False, type=bool)

    args = parser.parse_args()

    with open(args.config) as f:
        config_data = json.load(f)

    config = Config(config_data)
    setup_training_environment(args.config, "config.json", args.checkpoint_path)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    train(args, config)


if __name__ == "__main__":
    main()
