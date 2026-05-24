#!/usr/bin/env python3
"""
Generate all YAML configs for CMU-MOSEI hyperparameter sensitivity sweep.

Sweeps (per paper Table):
  MAD  (λ1): 0.0, 0.1, 0.3, 0.5, 0.7, 1.0   — λ2=0.0, λ3=0.0
  OT   (λ2): 0.00, 0.01, 0.03, 0.05, 0.10    — λ1=0.5, λ3=0.0
  Hyp  (λ3): 0.00, 0.01, 0.05, 0.10, 0.20    — λ1=0.5, λ2=0.0

Run:
  python3 scripts/sensitivity/generate_sensitivity_configs.py
"""
import os, yaml

ROOT    = ""
OUT_DIR = f"{ROOT}/configs/mosei/sensitivity"
RES_DIR = f"{ROOT}/experiments/results/mosei/sensitivity"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

DATA = {
    "train_manifest": f"{ROOT}/data/processed/mosei/mosei_utt_train.jsonl",
    "val_manifest":   f"{ROOT}/data/processed/mosei/mosei_utt_val.jsonl",
    "test_manifest":  f"{ROOT}/data/processed/mosei/mosei_utt_test.jsonl",
    "num_classes": 6,
    "task": "6class_multilabel",
    "csd_labels":  "/data/raw/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd",
    "csd_words":   "/data/raw/CMU-MOSEI/languages/CMU_MOSEI_TimestampedWordVectors.csd",
    "csd_audio":   "/data/raw/CMU-MOSEI/acoustics/CMU_MOSEI_COVAREP.csd",
    "csd_visual":  "/data/raw/CMU-MOSEI/visuals/CMU_MOSEI_VisualFacet42.csd",
    "max_text_len": 96,
}

MODEL = {
    "text_model_name": "roberta-base",
    "d_model": 256,
    "use_audio": True,
    "use_video": True,
    "modality_dropout": 0.1,
}

TRAIN = {
    "seed": 42,
    "epochs": 35,
    "batch_size_per_gpu": 32,
    "num_workers": 2,
    "lr": 2.0e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "amp": True,
    "patience": 10,
    "loss": "bce_sqrt",
}


def make_chado(w_mad, w_ot, w_hyp):
    return {
        "use_causal":     True,
        "use_hyperbolic": True,
        "use_ot":         True,
        "use_mad":        True,
        "w_causal":       0.005,
        "w_hyperbolic":   float(w_hyp),
        "w_ot":           float(w_ot),
        "w_mad":          float(w_mad),
        "mad_gamma":      0.5,
        "ot_sigma":       0.05,
        "ot_eps":         0.1,
        "ot_iters":       20,
    }


def write_cfg(name, w_mad, w_ot, w_hyp):
    cfg = {
        "data":  DATA,
        "model": MODEL,
        "chado": make_chado(w_mad, w_ot, w_hyp),
        "train": TRAIN,
        "logging": {
            "run_name": name,
            "out_dir":  f"{RES_DIR}/{name}",
        },
    }
    path = f"{OUT_DIR}/{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


configs = []

# ── MAD sweep (λ2=0.0, λ3=0.0) ───────────────────────────────────────────────
for v in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
    name = f"mad_{str(v).replace('.','')}"
    p = write_cfg(name, w_mad=v, w_ot=0.0, w_hyp=0.0)
    configs.append((name, "MAD", v, p))
    print(f"  [MAD  λ1={v}] → {p}")

# ── OT sweep (λ1=0.5, λ3=0.0) ────────────────────────────────────────────────
for v in [0.01, 0.03, 0.05, 0.10]:   # 0.00 already written as mad_05
    name = f"ot_{str(v).replace('.','')}"
    p = write_cfg(name, w_mad=0.5, w_ot=v, w_hyp=0.0)
    configs.append((name, "OT", v, p))
    print(f"  [OT   λ2={v}] → {p}")

# ── Hyperbolic sweep (λ1=0.5, λ2=0.0) ────────────────────────────────────────
for v in [0.01, 0.05, 0.10, 0.20]:   # 0.00 already written as mad_05
    name = f"hyp_{str(v).replace('.','')}"
    p = write_cfg(name, w_mad=0.5, w_ot=0.0, w_hyp=v)
    configs.append((name, "Hyp", v, p))
    print(f"  [Hyp  λ3={v}] → {p}")

print(f"\nTotal configs written: {len(configs)} (+ shared baseline mad_05)")
print(f"Config dir: {OUT_DIR}")

# Write manifest for runner
manifest = f"{OUT_DIR}/sweep_manifest.txt"
with open(manifest, "w") as f:
    for name, comp, val, path in configs:
        f.write(f"{name}\t{comp}\t{val}\t{path}\n")
    # Add baseline entry for OT λ2=0.00 and Hyp λ3=0.00 (shared with mad_05)
    f.write(f"mad_05\tOT\t0.0\t{OUT_DIR}/mad_05.yaml\n")
    f.write(f"mad_05\tHyp\t0.0\t{OUT_DIR}/mad_05.yaml\n")
print(f"Manifest : {manifest}")
