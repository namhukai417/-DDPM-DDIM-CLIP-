"""OG ↔ Syn 도메인 격차 측정 (CLIP feature space).

목적
----
"Syn 추가가 분류 성능에 도움이 되는가?"를 사후 해석하기 위한 보조 지표.
CLIP ViT-B/32 (laion2b_s34b_b79k) image encoder로 OG와 Syn 이미지를 임베딩하여,
클래스별 / 전체에 대해 다음 3종 지표를 산출한다.

1. **Centroid cosine distance** — 클래스별 OG·Syn 평균 임베딩의 1 − cos 유사도.
   0에 가까울수록 OG·Syn의 클래스 중심이 일치.
2. **MMD² (RBF kernel)** — 두 집합의 분포 차이. 0이면 분포 동일.
   sigma는 median heuristic.
3. **Linear separability AUC** — OG=0 / Syn=1 라벨로 logistic regression 5-fold CV AUC.
   0.5에 가까우면 두 도메인이 구분 불가(=닮음), 1.0에 가까우면 완전 분리(=다름).

Usage:
  python domain_gap.py [--device cuda:0] [--out domain_gap_v3.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import open_clip

ROOT = Path('/home/kyungho2/kh/project_CLIP/2차연구')
OG_DIR = ROOT / 'data/og_by_class'
SYN_DIR_OLD = ROOT / 'data/syn_filtered'
SYN_DIR_NEW = ROOT / 'data/syn_filtered_v3'
OUT_DEFAULT = ROOT / 'results_v3/domain_gap_v3.json'

LABELS_5 = ['A', 'C', 'V', 'C+V', 'others']
LABEL_DIRS = {'A': 'A', 'C': 'C', 'V': 'V', 'C+V': 'CV', 'others': 'others'}


def list_images_og(label):
    subdir = LABEL_DIRS[label]
    d = OG_DIR / subdir
    return sorted(d.glob('*.tif')) if d.is_dir() else []


def list_images_syn(label):
    out = []
    # 5클래스 모두 SYN_DIR_NEW(syn_filtered_v3)에서 적재 (A·V 포함, 동일 Quality-Plateau 필터)
    subdir = LABEL_DIRS[label]
    d = SYN_DIR_NEW / subdir
    if d.is_dir():
        out += sorted(d.glob('*.png')) + sorted(d.glob('*.tif'))
    return out


def embed_images(paths, model, preprocess, device, batch_size=32):
    """이미지 경로 리스트 → (N, D) L2-정규화 임베딩."""
    if not paths:
        return np.zeros((0, 512), dtype=np.float32)
    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = torch.stack([preprocess(Image.open(p).convert('RGB')) for p in batch_paths]).to(device)
            f = model.encode_image(imgs)
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def mmd_rbf(x, y, sigma=None):
    """MMD² with RBF kernel. sigma는 median heuristic."""
    if len(x) == 0 or len(y) == 0:
        return float('nan')
    xy = np.concatenate([x, y], axis=0)
    if sigma is None:
        # median heuristic on a subsample for speed
        sub_idx = np.random.default_rng(0).choice(len(xy), size=min(500, len(xy)), replace=False)
        sub = xy[sub_idx]
        sq = np.sum(sub ** 2, axis=1, keepdims=True)
        d2 = sq + sq.T - 2 * sub @ sub.T
        d2 = d2[np.triu_indices_from(d2, k=1)]
        sigma = max(np.sqrt(np.median(d2)) + 1e-8, 1e-3)
    g = 1.0 / (2.0 * sigma ** 2)

    def k(a, b):
        sq_a = np.sum(a ** 2, axis=1, keepdims=True)
        sq_b = np.sum(b ** 2, axis=1, keepdims=True)
        d2 = sq_a + sq_b.T - 2 * a @ b.T
        return np.exp(-g * np.clip(d2, 0, None))

    nx, ny = len(x), len(y)
    kxx = (k(x, x).sum() - np.trace(k(x, x))) / max(nx * (nx - 1), 1)
    kyy = (k(y, y).sum() - np.trace(k(y, y))) / max(ny * (ny - 1), 1)
    kxy = k(x, y).mean()
    return float(kxx + kyy - 2 * kxy)


def linear_separability(x, y, seed=0):
    """OG vs Syn 분류 5-fold CV AUC. 0.5=구분불가, 1.0=완전분리."""
    if len(x) < 5 or len(y) < 5:
        return float('nan')
    X = np.concatenate([x, y], axis=0)
    y_lab = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
    aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y_lab):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y_lab[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y_lab[te], prob))
    return float(np.mean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--out', type=str, default=str(OUT_DEFAULT))
    ap.add_argument('--syn-cap', type=int, default=500,
                    help='Syn 풀이 큰 경우 임베딩 비용 절감용 상한 (라벨당). 0=모두 사용')
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print('[load] OpenCLIP ViT-B/32 (laion2b_s34b_b79k)...')
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=args.device)

    results = {'labels': LABELS_5, 'per_class': {}, 'overall': {}}
    all_og, all_syn = [], []
    rng = np.random.default_rng(0)

    for lab in LABELS_5:
        og_paths = list_images_og(lab)
        syn_paths = list_images_syn(lab)
        if args.syn_cap > 0 and len(syn_paths) > args.syn_cap:
            syn_paths = list(rng.choice(syn_paths, size=args.syn_cap, replace=False))

        print(f'[embed] {lab}: OG={len(og_paths)} Syn={len(syn_paths)}')
        og_emb = embed_images([str(p) for p in og_paths], model, preprocess, args.device)
        syn_emb = embed_images([str(p) for p in syn_paths], model, preprocess, args.device)

        if len(og_emb) and len(syn_emb):
            og_c = og_emb.mean(axis=0)
            og_c = og_c / (np.linalg.norm(og_c) + 1e-8)
            syn_c = syn_emb.mean(axis=0)
            syn_c = syn_c / (np.linalg.norm(syn_c) + 1e-8)
            cos_d = float(1.0 - og_c @ syn_c)
        else:
            cos_d = float('nan')

        mmd2 = mmd_rbf(og_emb, syn_emb) if len(og_emb) and len(syn_emb) else float('nan')
        auc = linear_separability(og_emb, syn_emb) if len(og_emb) and len(syn_emb) else float('nan')

        results['per_class'][lab] = {
            'n_og': int(len(og_emb)),
            'n_syn': int(len(syn_emb)),
            'centroid_cos_dist': cos_d,
            'mmd2_rbf': mmd2,
            'linsep_auc_5cv': auc,
        }
        print(f'  → cos_d={cos_d:.4f}  MMD²={mmd2:.5f}  AUC={auc:.3f}')
        all_og.append(og_emb)
        all_syn.append(syn_emb)

    all_og = np.concatenate([x for x in all_og if len(x)], axis=0) if any(len(x) for x in all_og) else np.zeros((0, 512))
    all_syn = np.concatenate([x for x in all_syn if len(x)], axis=0) if any(len(x) for x in all_syn) else np.zeros((0, 512))

    overall_cos = float('nan')
    if len(all_og) and len(all_syn):
        og_c = all_og.mean(axis=0)
        og_c = og_c / (np.linalg.norm(og_c) + 1e-8)
        syn_c = all_syn.mean(axis=0)
        syn_c = syn_c / (np.linalg.norm(syn_c) + 1e-8)
        overall_cos = float(1.0 - og_c @ syn_c)

    overall_mmd = mmd_rbf(all_og, all_syn) if len(all_og) and len(all_syn) else float('nan')
    overall_auc = linear_separability(all_og, all_syn) if len(all_og) and len(all_syn) else float('nan')

    results['overall'] = {
        'n_og': int(len(all_og)),
        'n_syn': int(len(all_syn)),
        'centroid_cos_dist': overall_cos,
        'mmd2_rbf': overall_mmd,
        'linsep_auc_5cv': overall_auc,
    }
    print(f'\n[overall] cos_d={overall_cos:.4f}  MMD²={overall_mmd:.5f}  AUC={overall_auc:.3f}')

    with open(out_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
