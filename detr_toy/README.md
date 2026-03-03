# Toy DETR Starter (No PyTorch Required)

This folder is a lightweight DETR-style project you can run immediately with the packages available in this environment.

It keeps the key ideas:
- fixed query slots (`num_queries`)
- one-to-one Hungarian matching to build supervision
- explicit `no-object` slots
- count + localization evaluation

It is intentionally small and educational, not a production detector.

## Files

- `toy_dataset.py`: synthetic image + box generation
- `matcher.py`: Hungarian matching, IoU, precision/recall/F1
- `model.py`: fixed-slot MLP detector
- `train.py`: end-to-end training, evaluation, and qualitative outputs

## Run

```bash
python3 detr_toy/train.py --warmup-epochs 30 --match-epochs 10 --n-train 800 --n-val 200
```

Quick smoke test:

```bash
python3 detr_toy/train.py --warmup-epochs 4 --match-epochs 2 --n-train 120 --n-val 40
```

## Outputs

Saved under `detr_toy/artifacts/`:
- `toy_set_detector.pkl`
- `history.json`
- `config.json`
- `qualitative/pred_*.png` (green = GT, red = predicted)

## Next step to move closer to real DETR

When network/package install is available, switch to PyTorch and replace `model.py` with:
1. CNN backbone + transformer encoder/decoder
2. learned object queries
3. Hungarian loss with class CE + box L1/GIoU
