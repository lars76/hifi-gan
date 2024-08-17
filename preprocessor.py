from pathlib import Path
import torch
import tgt
from tqdm import tqdm

MODEL_PATH = "/home/nieradzik/speech/FINAL4/lightspeech/more_data2/model.pt"
DATASET_PATH = Path("/home/SSD/new_location/")
DEVICE = "cuda:3"


def extract_text_from_tiers(textgrid_path, tier_name):
    tg = tgt.read_textgrid(textgrid_path)
    tier = tg.get_tier_by_name(tier_name)
    return [interval.text for interval in tier.intervals if interval.text]


def load_model(state_dict):
    if "lightspeech" in MODEL_PATH:
        from lightspeech import Model
    else:
        from fastspeech2 import Model
    model = (
        Model(
            num_phones=state_dict["num_phones"],
            num_speakers=state_dict["num_speakers"],
            num_mel_bins=state_dict["num_mel_bins"],
            d_model=state_dict.get("d_model", 256),
        )
        .to(DEVICE)
        .eval()
    )
    model.load_state_dict(state_dict["state_dict"], strict=True)
    return model


def process_text(text, pinyin_dict, phone_dict):
    tokens = [phone_dict["<sil>"]]
    tones = [1]

    for pinyin in text:
        phonemes = pinyin_dict[pinyin[:-1]].split()
        tokens.extend(phone_dict[p] for p in phonemes)
        tones.extend([(int(pinyin[-1]) + 1)] * len(phonemes))

    tokens.append(phone_dict["<sil>"])
    tones.append(1)

    return tokens, tones


def main():
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model = load_model(state_dict)
    speaker_mapper = {k["name"]: k["speaker_id"] for k in state_dict["speaker_dict"]}
    pinyin_dict, phone_dict = state_dict["pinyin_dict"], state_dict["phone_dict"]

    textgrid_files = sorted(DATASET_PATH.glob("*/*.TextGrid"))
    with torch.inference_mode():
        for textgrid_filename in tqdm(textgrid_files, desc="Processing files"):
            text = extract_text_from_tiers(textgrid_filename, "pinyins")
            tokens, tones = process_text(text, pinyin_dict, phone_dict)
            speaker_id = speaker_mapper[textgrid_filename.parent.name]

            token_ids = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
            tone_ids = torch.tensor([tones], dtype=torch.long).to(DEVICE)
            speaker_id = torch.tensor([speaker_id], dtype=torch.long).to(DEVICE)

            mel, *_ = model(speaker_id, token_ids, tone_ids)
            torch.save(mel, textgrid_filename.with_suffix(".pt"))


if __name__ == "__main__":
    main()
