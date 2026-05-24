"""
Pre-compute dialogue context embeddings for IEMOCAP.

For each utterance u_i in dialogue d:
  context_emb[u_i] = mean(text_emb[u_0], ..., text_emb[u_{i-1}])
                   = zeroes if i == 0 (first turn has no prior context)

Saves: data/cache/iemocap_context_embs.npz
  keys: utt_ids (array of str), embs (array [N, 768])

Run:
  CUDA_VISIBLE_DEVICES=5 python3 scripts/preprocess/compute_context_embs_iemocap.py
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, '/home/tahirahmad/Project_Code/CHADO_EMNLP')
BASE = '/home/tahirahmad/Project_Code/CHADO_EMNLP'

MODEL_NAME = 'roberta-base'
BATCH_SIZE = 64
MAX_LEN    = 96
OUT_PATH   = f'{BASE}/data/cache/iemocap_context_embs.npz'
os.makedirs(f'{BASE}/data/cache', exist_ok=True)

SPLITS = [
    f'{BASE}/data/manifests/iemocap/train.csv',
    f'{BASE}/data/manifests/iemocap/val.csv',
    f'{BASE}/data/manifests/iemocap/test.csv',
]


def _parse_dial_turn(utt_id):
    import re
    m = re.match(r'^(.*?)_[MF](\d+)$', utt_id)
    if m:
        return m.group(1), int(m.group(2))
    return utt_id, 0


class TextDataset(Dataset):
    def __init__(self, utt_ids, texts, tok):
        self.utt_ids = utt_ids
        self.enc = tok(
            texts, padding='max_length', truncation=True,
            max_length=MAX_LEN, return_tensors='pt'
        )
    def __len__(self): return len(self.utt_ids)
    def __getitem__(self, i):
        return {
            'input_ids':      self.enc['input_ids'][i],
            'attention_mask': self.enc['attention_mask'][i],
        }


def encode_all(utt_ids, texts, model, tok, device):
    ds = TextDataset(utt_ids, texts, tok)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    all_embs = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            iids = batch['input_ids'].to(device)
            amsk = batch['attention_mask'].to(device)
            out  = model(input_ids=iids, attention_mask=amsk)
            cls  = out.last_hidden_state[:, 0, :]   # [B, 768]
            cls  = F.normalize(cls, dim=1)
            all_embs.append(cls.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)   # [N, 768]


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    tok   = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # ── collect all utterances across all splits ──────────────────────────
    dfs = [pd.read_csv(p) for p in SPLITS if os.path.exists(p)]
    df  = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='utt_id')
    print(f'Total unique utterances: {len(df)}')

    utt_ids = df['utt_id'].tolist()
    texts   = df['transcript'].fillna('').astype(str).tolist()

    # ── parse dialogue structure ───────────────────────────────────────────
    dial_turn = [_parse_dial_turn(u) for u in utt_ids]
    df['_dial'] = [x[0] for x in dial_turn]
    df['_turn'] = [x[1] for x in dial_turn]
    df = df.reset_index(drop=True)

    # ── encode all utterances ──────────────────────────────────────────────
    print('Encoding utterances...')
    embs = encode_all(utt_ids, texts, model, tok, device)   # [N, 768]
    utt2emb = {uid: embs[i] for i, uid in enumerate(utt_ids)}

    # ── build context embeddings ───────────────────────────────────────────
    # For each utterance, context_emb = mean of prior turns in same dialogue.
    # First turn → zero vector (no prior context).
    print('Building context embeddings...')
    ctx_embs = np.zeros_like(embs)   # [N, 768]

    # group by dialogue, sort by turn
    uid2idx = {uid: i for i, uid in enumerate(utt_ids)}
    for dial_id, grp in df.groupby('_dial'):
        grp_sorted = grp.sort_values('_turn')
        uids_sorted = grp_sorted['utt_id'].tolist()
        running_sum = np.zeros(768, dtype=np.float64)
        for k, uid in enumerate(uids_sorted):
            idx = uid2idx[uid]
            if k == 0:
                ctx_embs[idx] = 0.0   # first turn: zero context
            else:
                ctx_embs[idx] = (running_sum / k).astype(np.float32)
            running_sum += utt2emb[uid].astype(np.float64)

    # ── save ──────────────────────────────────────────────────────────────
    np.savez_compressed(
        OUT_PATH,
        utt_ids=np.array(utt_ids, dtype=object),
        embs=ctx_embs.astype(np.float32),
    )
    print(f'Saved → {OUT_PATH}  shape={ctx_embs.shape}')
    # quick sanity
    n_nonzero = (np.abs(ctx_embs).sum(axis=1) > 0).sum()
    print(f'Non-zero context embs: {n_nonzero}/{len(utt_ids)} '
          f'(first turns in each dialogue are zeros)')


if __name__ == '__main__':
    main()
