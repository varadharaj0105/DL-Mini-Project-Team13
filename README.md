# Field-Robust Paddy and Crop Disease Recognition with Lightweight Vision Transformers for Edge Deployment

Deep Learning mini project (course 23I006), PSG College of Technology, Department of Information Technology.
Team 13: Chandru K (23I315), Soma Prasanth R (23I362), Sriram R (23I365), Varadharaj R (23I369).

## What this project does
A MobileNetV3-Large student is distilled from a lightweight vision-transformer teacher (MobileViT-S for PlantDoc, EfficientFormer-L1 for Paddy). The teacher sees the clean image; the student sees a distorted, field-style view. We measure where this helps (PlantDoc) and where it does not (unseen rice varieties), with grouped splits that keep near-duplicate photographs together.

## Headline results (macro-F1 %, test set)
| Setting | MobileNetV3 baseline | Ours (distillation + field aug) | Teacher |
|---|---|---|---|
| PlantDoc, 3 seeds | 56.0 ± 1.7 | 62.1 ± 1.0 | 71.1 (MobileViT-S) |
| Paddy, in-distribution (seed 42) | 93.4 | 89.8 | 95.1 (EfficientFormer-L1) |
| Paddy, unseen varieties (seed 42) | 31.5 | 29.3 | 34.5 (EfficientFormer-L1) |

Distillation helps on PlantDoc; no model survives a change of rice variety. Adding MixStyle (feature-statistic mixing) or tuning the distillation weight α gave no reliable further gain: PlantDoc 62.9 ± 1.8 (MixStyle, α = 0.5) and 62.1 ± 2.2 (α = 0.7) against 61.1 ± 1.1 for plain distillation, and 30–32 macro-F1 for the student under variety shift (baseline 31.5). See the paper for all tables and the Grad-CAM analysis.

## Layout
```
src/prepare_data.py   download datasets, build pHash-grouped splits
src/train.py          baselines (MobileNetV3-Large, MobileViT-S, EfficientFormer-L1)
src/train_kd.py       field augmentation + asymmetric distillation (used for the paper tables)
src/train_nov.py      same + MixStyle option and cached teacher logits (faster)
src/gradcam.py        Grad-CAM figure
configs/experiments.yaml   all hyper-parameters; configs/splits/*.csv   the exact split files used
tests/                leakage and unit tests (pytest -q)
```

## Reproduce
```bash
pip install -r requirements.txt
python src/prepare_data.py paddy    --root data/paddy      # 10,407-image Kaggle subset of Paddy Doctor (public HF mirror)
python src/prepare_data.py plantdoc --root data/plantdoc
# (the paper resized images to longest side 320 px for Paddy and 384 px for PlantDoc before training)
# or copy configs/splits/*_split.csv to <root>/split.csv to use our exact splits

# baselines
python src/train.py --root data/plantdoc_384 --model mobilenetv3_large_100 --epochs 12
python src/train.py --root data/plantdoc_384 --model mobilevit_s --epochs 12
python src/train.py --root data/plantdoc_384 --model efficientformer_l1 --epochs 12
# distillation + field augmentation (needs the teacher checkpoint produced above)
python src/train_kd.py --root data/plantdoc_384 --epochs 12 --field_aug \
  --teacher mobilevit_s --teacher_ckpt runs/plantdoc_384_mobilevit_s_seed42.pt
# variety shift (Paddy): add --variety_shift (train on ADT45, test on the other varieties)
# faster distillation (cached teacher logits) with optional MixStyle and alpha
python src/train_nov.py --root data/plantdoc_384 --epochs 12 --mixstyle 0.5 --alpha 0.5 \
  --teacher mobilevit_s --teacher_ckpt runs/plantdoc_384_mobilevit_s_seed42.pt
pytest -q
```
Seeds 42, 43, 44 for the PlantDoc ablation (`--seed`). Training used a single Colab T4 GPU.

## Limitations
10,407-image subset of Paddy Doctor (the full 16,225-image release needs an IEEE DataPort subscription); PlantDoc test set has 236 images; Paddy and variety-shift runs are single-seed; CPU latency is from a shared Colab CPU, not a phone.
