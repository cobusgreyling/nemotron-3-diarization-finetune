"""Fine-tune the Nemotron 3 Diarization speaker head on scripted overlaps.

The encoder stays frozen. Labels are RTTM files: who spoke when, including
overlap. Speakers are assigned to the model's eight channels by first arrival.
"""

from __future__ import annotations

import json
import random
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
MODEL_ID = "nvidia/Nemotron-3-Diarization"
SR = 16000
HOP_S = 0.01
N_SPK = 8
SEED = 42
LR = 1e-4
EPOCHS = 12
THRESHOLD = 0.5

VOICES = {"alice": "Alice", "daniel": "Daniel", "anna": "Anna"}
UTTERANCES = {
    "alice": [
        "The launch review starts now.",
        "I want the notes from yesterday.",
        "We ship on Friday if the tests pass.",
        "Hold on, that number looks wrong.",
    ],
    "daniel": [
        "I already sent the deck.",
        "The overlap is the whole problem.",
        "Give me one minute with the logs.",
        "Yes, I can take that action.",
    ],
    "anna": [
        "Customers heard both of you at once.",
        "Label the speakers before you train.",
        "This clip has three voices.",
        "I will check the holdout set.",
    ],
}

# overlap is how many seconds this turn starts before the previous turn ends.
SCENES = [
    ("train", "tr01", [("alice", 0, 0.0), ("daniel", 0, 0.7), ("anna", 0, 0.5)]),
    ("train", "tr02", [("daniel", 0, 0.0), ("alice", 1, 0.4)]),
    ("train", "tr03", [("anna", 0, 0.0), ("daniel", 1, 0.9), ("alice", 0, 0.3)]),
    ("train", "tr04", [("alice", 1, 0.0), ("anna", 1, 1.0)]),
    ("train", "tr05", [("daniel", 1, 0.0), ("anna", 0, 0.2), ("alice", 1, 0.8)]),
    ("train", "tr06", [("anna", 1, 0.0), ("alice", 0, 0.6)]),
    ("train", "tr07", [("alice", 0, 0.0), ("anna", 0, 0.0), ("daniel", 0, 0.5)]),
    ("train", "tr08", [("daniel", 0, 0.0), ("alice", 0, 1.1), ("anna", 1, 0.4)]),
    ("train", "tr09", [("anna", 0, 0.0), ("daniel", 0, 0.35)]),
    ("train", "tr10", [("alice", 1, 0.0), ("daniel", 1, 0.75), ("anna", 1, 0.55)]),
    ("dev", "dv01", [("alice", 2, 0.0), ("daniel", 2, 0.6)]),
    ("dev", "dv02", [("anna", 2, 0.0), ("alice", 2, 0.85), ("daniel", 2, 0.4)]),
    ("holdout", "ho01", [("daniel", 3, 0.0), ("alice", 3, 0.7), ("anna", 3, 0.5)]),
    ("holdout", "ho02", [("anna", 3, 0.0), ("daniel", 3, 0.45)]),
    ("holdout", "ho03", [("alice", 3, 0.0), ("anna", 3, 1.0)]),
    ("holdout", "ho04", [("daniel", 3, 0.0), ("anna", 3, 0.3), ("alice", 3, 0.8)]),
]


def der_counts(ref: np.ndarray, hyp: np.ndarray) -> dict[str, int]:
    """Frame DER with overlap counted and speaker ids already aligned.

    A reference speaker-frame matched to the same channel is correct. An
    unmatched reference speaker-frame paired with an unmatched hypothesis
    channel is a speaker error. Leftover reference is a miss. Leftover
    hypothesis is a false alarm.
    """
    ref = np.asarray(ref, dtype=bool)
    hyp = np.asarray(hyp, dtype=bool)
    if ref.shape != hyp.shape:
        raise ValueError(f"shape {ref.shape} != {hyp.shape}")
    miss = fa = speaker_error = 0
    for t in range(ref.shape[0]):
        ref_ids = set(np.flatnonzero(ref[t]).tolist())
        hyp_ids = set(np.flatnonzero(hyp[t]).tolist())
        correct = ref_ids & hyp_ids
        ref_left = len(ref_ids - correct)
        hyp_left = len(hyp_ids - correct)
        paired = min(ref_left, hyp_left)
        speaker_error += paired
        miss += ref_left - paired
        fa += hyp_left - paired
    return {
        "total": int(ref.sum()),
        "miss": miss,
        "false_alarm": fa,
        "speaker_error": speaker_error,
    }


def der_rate(counts: dict[str, int]) -> float | None:
    if counts["total"] == 0:
        return None
    return (counts["miss"] + counts["false_alarm"] + counts["speaker_error"]) / counts["total"]


def _check_der() -> None:
    def bits(*rows: str) -> np.ndarray:
        return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)

    perfect = der_counts(bits("10", "11"), bits("10", "11"))
    assert der_rate(perfect) == 0.0
    missed = der_counts(bits("10"), bits("00"))
    assert missed["miss"] == 1 and der_rate(missed) == 1.0
    swap = der_counts(bits("10"), bits("01"))
    assert swap["speaker_error"] == 1 and swap["miss"] == 0 and der_rate(swap) == 1.0
    overlap = der_counts(bits("11"), bits("10"))
    assert overlap["miss"] == 1 and overlap["speaker_error"] == 0 and der_rate(overlap) == 0.5
    extra = der_counts(bits("10"), bits("11"))
    assert extra["false_alarm"] == 1 and der_rate(extra) == 1.0


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        if handle.getframerate() != SR or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16 kHz mono 16-bit")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path, audio: np.ndarray) -> None:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def synthesize_utterances() -> dict[tuple[str, int], np.ndarray]:
    cache = DATA / "utterances"
    cache.mkdir(parents=True, exist_ok=True)
    clips = {}
    for speaker, lines in UTTERANCES.items():
        for index, text in enumerate(lines):
            path = cache / f"{speaker}-{index}.wav"
            if not path.exists():
                subprocess.run(
                    ["say", "-v", VOICES[speaker], "-o", str(path), "--data-format=LEI16@16000", text],
                    check=True,
                )
            clips[(speaker, index)] = read_wav(path)
    return clips


def build_scene(clips, turns) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    cursor = int(0.3 * SR)
    spans = []
    for speaker, index, overlap in turns:
        clip = clips[(speaker, index)]
        start = max(int(0.15 * SR), cursor - int(round(overlap * SR)))
        spans.append((speaker, clip, start))
        cursor = max(cursor, start + len(clip))
    total = cursor + int(0.3 * SR)
    mix = np.zeros(total, dtype=np.float32)
    segments = []
    for speaker, clip, start in spans:
        end = start + len(clip)
        mix[start:end] += clip
        segments.append((speaker, start / SR, len(clip) / SR))
    return mix, segments


def write_rttm(path: Path, file_id: str, segments: list[tuple[str, float, float]]) -> None:
    lines = []
    for speaker, start, duration in segments:
        lines.append(
            f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def prepare() -> list[dict]:
    clips = synthesize_utterances()
    rows = []
    for split, file_id, turns in SCENES:
        audio, segments = build_scene(clips, turns)
        wav_path = DATA / split / f"{file_id}.wav"
        rttm_path = DATA / "rttm" / f"{file_id}.rttm"
        write_wav(wav_path, audio)
        write_rttm(rttm_path, file_id, segments)
        speakers = []
        for speaker, _, _ in segments:
            if speaker not in speakers:
                speakers.append(speaker)
        rows.append(
            {
                "id": file_id,
                "split": split,
                "audio_filepath": str(wav_path.relative_to(ROOT)),
                "offset": 0,
                "duration": round(len(audio) / SR, 3),
                "label": "infer",
                "text": "-",
                "num_speakers": len(speakers),
                "rttm_filepath": str(rttm_path.relative_to(ROOT)),
                "arrival": speakers_by_arrival(segments),
            }
        )
    for split in ("train", "dev", "holdout"):
        subset = [row for row in rows if row["split"] == split]
        (DATA / f"{split}_manifest.json").write_text("".join(json.dumps(row) + "\n" for row in subset))
    (DATA / "manifest.json").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def speakers_by_arrival(segments: list[tuple[str, float, float]]) -> list[str]:
    first = {}
    for speaker, start, _ in segments:
        first[speaker] = min(start, first.get(speaker, start))
    return [speaker for speaker, _ in sorted(first.items(), key=lambda item: (item[1], item[0]))]


def load_manifest(split: str | None = None) -> list[dict]:
    path = DATA / ("manifest.json" if split is None else f"{split}_manifest.json")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_rttm(path: Path) -> list[tuple[str, float, float]]:
    segments = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        segments.append((parts[7], float(parts[3]), float(parts[4])))
    return segments


def target_matrix(row: dict, n_frames: int) -> np.ndarray:
    target = np.zeros((n_frames, N_SPK), dtype=np.float32)
    segments = read_rttm(ROOT / row["rttm_filepath"])
    arrival = speakers_by_arrival(segments)
    channel = {speaker: index for index, speaker in enumerate(arrival)}
    for speaker, start, duration in segments:
        begin = int(round(start / HOP_S))
        end = int(round((start + duration) / HOP_S))
        begin = min(max(begin, 0), n_frames)
        end = min(max(end, 0), n_frames)
        target[begin:end, channel[speaker]] = 1.0
    return target


def load_model(device: torch.device):
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor
    from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import (
        Nemotron3DiarizationSpeakerCache,
    )

    original = Nemotron3DiarizationSpeakerCache.update

    def update_without_grad(self, *args, **kwargs):
        # The cache is streaming state. Keeping it out of the graph stops the
        # in-place cache write from breaking backward through the logits.
        with torch.no_grad():
            return original(self, *args, **kwargs)

    Nemotron3DiarizationSpeakerCache.update = update_without_grad

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.to(device)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier.")
    original_forward = model.classifier.forward

    def classifier_on_detached(hidden_states):
        return original_forward(hidden_states.detach())

    model.classifier.forward = classifier_on_detached
    return processor, model


def logits_for(processor, model, audio: np.ndarray, device: torch.device) -> tuple[torch.Tensor, int]:
    inputs = processor(audio, sampling_rate=SR, return_tensors="pt")
    frame_inputs = {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}
    outputs = model(**frame_inputs)
    logits = outputs.logits[0]
    mask = inputs["attention_mask"][0]
    valid = int(mask.sum().item()) if mask is not None else logits.shape[0]
    valid = min(valid, logits.shape[0])
    return logits[:valid], valid


@torch.no_grad()
def score_split(processor, model, split: str, device: torch.device) -> dict:
    rows = load_manifest(split)
    totals = {"total": 0, "miss": 0, "false_alarm": 0, "speaker_error": 0}
    files = []
    for row in rows:
        audio = read_wav(ROOT / row["audio_filepath"])
        logits, valid = logits_for(processor, model, audio, device)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        hyp = probs >= THRESHOLD
        ref = target_matrix(row, valid) >= 0.5
        counts = der_counts(ref, hyp)
        for key in totals:
            totals[key] += counts[key]
        files.append({"id": row["id"], "der": der_rate(counts), "duration": row["duration"], **counts})
    return {"split": split, "der": der_rate(totals), "files": files, **totals}


def train(processor, model, device: torch.device) -> dict:
    head = [param for param in model.classifier.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(head, lr=LR, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    train_rows = load_manifest("train")
    history = []
    best = None
    OUT.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        random.Random(SEED + epoch).shuffle(train_rows)
        model.classifier.train()
        losses = []
        for row in train_rows:
            audio = read_wav(ROOT / row["audio_filepath"])
            logits, valid = logits_for(processor, model, audio, device)
            target = torch.tensor(target_matrix(row, valid), device=device)
            loss = loss_fn(logits.float(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.classifier.eval()
        dev = score_split(processor, model, "dev", device)
        mean_loss = float(np.mean(losses))
        print(f"epoch {epoch:02d}  loss {mean_loss:.4f}  dev DER {dev['der']:.4f}", flush=True)
        state = {key: value.detach().cpu().clone() for key, value in model.classifier.state_dict().items()}
        record = {"epoch": epoch, "loss": mean_loss, "dev_der": dev["der"]}
        history.append(record)
        if best is None or dev["der"] < best["dev_der"]:
            best = {**record, "state": state}
    model.classifier.load_state_dict(best["state"])
    torch.save(best["state"], OUT / "classifier.pt")
    return {"lr": LR, "epochs": EPOCHS, "best_epoch": best["epoch"], "history": history}


def apply_checkpoint(model, path: Path) -> None:
    state = torch.load(path, map_location="cpu")
    model.classifier.load_state_dict(state)


def main() -> None:
    _check_der()
    torch.manual_seed(SEED)
    random.seed(SEED)
    if not (DATA / "manifest.json").exists():
        prepare()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {device}", flush=True)
    processor, model = load_model(device)
    before = {split: score_split(processor, model, split, device) for split in ("train", "dev", "holdout")}
    print(
        "base DER  "
        + "  ".join(f"{split} {before[split]['der']:.4f}" for split in before),
        flush=True,
    )
    training = train(processor, model, device)
    after = {split: score_split(processor, model, split, device) for split in ("train", "dev", "holdout")}
    print(
        "tuned DER "
        + "  ".join(f"{split} {after[split]['der']:.4f}" for split in after),
        flush=True,
    )
    # Drop tensors before writing the report.
    training = {key: value for key, value in training.items()}
    report = {
        "model": MODEL_ID,
        "device": str(device),
        "threshold": THRESHOLD,
        "hop_seconds": HOP_S,
        "collar_seconds": 0,
        "ignore_overlap": False,
        "frozen": "encoder",
        "trained": "classifier",
        "train_clips": 10,
        "dev_clips": 2,
        "holdout_clips": 4,
        "before": before,
        "after": after,
        "training": training,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
