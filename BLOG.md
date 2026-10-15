# Fine-tuning Nemotron 3 Diarization

The public model is [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization). It is a 99M Sortformer. It answers who spoke when, including overlap, for up to eight speakers. It does not transcribe.

This Mac is an M2. It has no CUDA, so the encoder stayed frozen and the speaker head was trained here, on Apple MPS.

## 1. Domain and labels

Three Mac voices: Alice, Daniel, and Anna. Sixteen clips, each between 3.5 and 5.6 seconds, with scripted overlap. Ten are train, two are dev, four are holdout. The holdout sentences do not appear in the train clips.

An RTTM line is one stretch of one speaker.

```
SPEAKER tr01 1 0.300 1.518 <NA> <NA> alice <NA> <NA>
```

Alice starts at 0.300 seconds and continues for 1.518 seconds. Overlap is two lines that cover the same time.

The manifest line is the same shape NeMo expects.

```json
{"audio_filepath": "data/train/tr01.wav", "offset": 0, "duration": 4.395, "label": "infer", "text": "-", "num_speakers": 3, "rttm_filepath": "data/rttm/tr01.rttm"}
```

The first voice to speak is channel 0. The next new voice is channel 1. That is arrival order, which is how this model names speakers.

## 2. Score the base model first

DER is missed speech, plus false alarm, plus speaker error, divided by the reference speech. The collar is 0. Overlap counts. Frames are 10 ms. A frame counts as speech when its probability is at least 0.5.

| Split | Clips | Base DER |
|---|---:|---:|
| Train | 10 | 11.68% |
| Dev | 2 | 2.22% |
| Holdout | 4 | 2.74% |

Almost all of the train error sits in two clips, `tr03` (35.0%) and `tr10` (39.8%). The holdout has no speaker errors.

## 3. Fine-tune

The encoder stays frozen. AdamW trains the speaker head at learning rate `1e-4`, with binary cross-entropy, for 12 epochs. Epoch 6 had the lowest dev DER, so that head is the one that was scored.

```bash
python finetune.py
```

On a CUDA machine, with hours of your own RTTM, NeMo Speech trains the full checkpoint. Use the 8-speaker streaming config. The 4-speaker yaml does not match this head.

```bash
python examples/speaker_tasks/diarization/neural_diarizer/streaming_sortformer_diar_train.py \
  --config-path=../conf/neural_diarizer \
  --config-name=sortformer_streaming_8spk.yaml \
  trainer.devices=1 \
  trainer.precision=bf16 \
  trainer.max_epochs=5 \
  model.train_ds.manifest_filepath=/path/train_manifest.json \
  model.validation_ds.manifest_filepath=/path/val_manifest.json \
  model.optim.lr=1e-5 \
  +init_from_nemo_model=/path/Nemotron-3-Diarization.nemo
```

## 4. Result

| Split | Base DER | Tuned DER |
|---|---:|---:|
| Train | 11.68% | 7.67% |
| Dev | 2.22% | 1.82% |
| Holdout | 2.74% | 2.38% |

Holdout, 1,972 reference frames:

| | Miss | False alarm | Speaker error |
|---|---:|---:|---:|
| Base | 18 | 36 | 0 |
| Tuned | 11 | 36 | 0 |

The head recovered 7 missed frames on the holdout and left the false alarms where they were. `tr10` moved from 39.8% to 10.7%. `tr03` stayed at 34.7%.

The saved head is `outputs/classifier.pt`. The base weights stay on Hugging Face. This code is MIT. The model is [OpenMDW 1.1](https://huggingface.co/nvidia/Nemotron-3-Diarization).
