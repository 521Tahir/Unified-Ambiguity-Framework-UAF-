

## Environment Setup

```bash
conda env create -f environment.yml
conda activate name
```

Or via pip:
```bash
pip install -r requirements.txt
```

---

## Data Preparation

### IEMOCAP
```bash
python scripts/preprocess/cache_features_iemocap.py \
  --data-root /path/to/IEMOCAP_full_release \
  --out-dir data/cache/iemocap
```

### CMU-MOSEI
```bash
python scripts/preprocess/build_mosei_utt_manifests.py \
  --csd-root /path/to/CMU-MOSEI \
  --out-dir data/processed/mosei
```

### MELD
```bash
python scripts/preprocess/build_meld_csv.py \
  --data-root /path/to/MELD \
  --out-dir data/processed/meld
```

---

## Training

### IEMOCAP (single GPU)
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 \
  scripts/train/train_iemocap.py \
  --config configs/iemocap/proxy_20ep.yaml \
  --proxy mad \
  --out-dir experiments/results/iemocap/proxy_comparison/mad/seed_42
```

### MELD (single GPU)
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 \
  scripts/train/train_meld.py \
  --config configs/meld/proxy_20ep.yaml \
  --proxy mad \
  --out-dir experiments/results/meld/proxy_comparison/mad/seed_42
```

### CMU-MOSEI (single GPU)
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29502 \
  scripts/train/train_mosei.py \
  --config configs/mosei/proxy_20ep.yaml \
  --proxy mad \
  --out-dir experiments/results/mosei/proxy_comparison/mad/seed_42




## Ablation Studies

```bash
# Component ablations (w/o MAD, w/o OT, w/o Radial)
bash scripts/ablations/run_component_ablations.sh

# Latent factor ablations (w/o z_c, z_u, z_t, z_m, z_e)
bash scripts/ablations/run_ablations_iemocap.sh
```

---

## Evaluation

```bash
# Ambiguity stratification analysis
python scripts/eval/run_A1_ambiguity_stratification.py

# OT perturbation verification (Prop. 1)
python scripts/eval/run_A5_ot_perturbation.py

# Statistical significance tests
python evaluation/significance/run_significance_full.py

# Generate all paper figures
python scripts/plots/generate_paper_figures.py
```

## License

Anonymous submission — license to be specified upon de-anonymization.
