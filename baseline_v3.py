"""V3 베이스라인 — zero-shot CLIP / linear probe / majority class.

목적
----
fine-tune 결과(V3-Ctrl/A/B/C)와의 비교 기준선 4종.

1. **majority**           : 학습셋 최빈 클래스를 항상 예측 (floor)
2. **clip_zeroshot**       : 학습 없이 CLIP + prompt bank anchor로 분류
3. **clip_linprobe_og**    : CLIP feature 고정 → OG train만으로 logistic regression
4. **clip_linprobe_og+syn**: CLIP feature 고정 → OG train + V3-A 비율 Syn으로 logistic regression
                              (linear probe로 본 "Syn 추가 효과"의 상한선 추정)

manifest의 (case=100=V3-Ctrl/case=101=V3-A) split을 그대로 사용해
fine-tune 결과와 동일한 train/test 위에서 비교한다.

Usage:
  python baseline_v3.py [--device cuda:0] [--out results_v3/baselines.json]
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix,
    precision_recall_fscore_support,
)
import open_clip

ROOT = Path('/home/kyungho2/kh/project_CLIP/2차연구')
MANIFEST = ROOT / 'data/case_manifests/v3_cases.csv'
OUT_DEFAULT = ROOT / 'results_v3/baselines.json'

LABELS_5 = ['A', 'C', 'V', 'C+V', 'others']

PROMPT_BANK = {
    'A': [
        'an SEM image of needle-like aragonite crystals',
        'a scanning electron micrograph showing elongated acicular aragonite needles',
        'FE-SEM image of fibrous aragonite polymorph of calcium carbonate',
        'an electron microscopy image of rod-shaped aragonite crystals',
        'SEM image showing acicular CaCO3 aragonite phase with needle morphology',
    ],
    'C': [
        'an SEM image of cubic calcite crystals',
        'a scanning electron micrograph showing rhombohedral calcite blocks',
        'FE-SEM image of euhedral calcite cubes of calcium carbonate',
        'an electron microscopy image of well-defined cubic calcite crystals',
        'SEM image showing rhombohedral CaCO3 calcite phase with cubic morphology',
    ],
    'V': [
        'an SEM image of spherical vaterite particles',
        'a scanning electron micrograph showing rounded vaterite spherules',
        'FE-SEM image of porous spherical vaterite of calcium carbonate',
        'an electron microscopy image of clustered vaterite spheres',
        'SEM image showing spheroidal CaCO3 vaterite phase with spherical morphology',
    ],
    'C+V': [
        'an SEM image showing both cubic calcite and spherical vaterite crystals',
        'a scanning electron micrograph of mixed calcite and vaterite phases of calcium carbonate',
        'FE-SEM image of co-existing rhombohedral calcite and spheroidal vaterite',
        'an electron microscopy image of CaCO3 with both calcite cubes and vaterite spheres',
        'SEM image showing C-V polymorph mixture of cubic calcite and spherical vaterite particles',
    ],
    'others': [
        'an SEM image not clearly belonging to a single CaCO3 polymorph',
        'a scanning electron micrograph of mixed-phase or unclassifiable calcium carbonate',
        'FE-SEM image of a low-quality or ambiguous CaCO3 sample',
        'an electron microscopy image of CaCO3 with no dominant single polymorph morphology',
        'SEM image not assignable to aragonite, calcite, or vaterite alone',
    ],
}


def embed_batch(paths, model, preprocess, device, batch_size=32):
    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            bp = paths[i:i + batch_size]
            imgs = torch.stack([preprocess(Image.open(p).convert('RGB')) for p in bp]).to(device)
            f = model.encode_image(imgs)
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512))


def metrics_block(true, pred, labels=LABELS_5):
    acc = float(accuracy_score(true, pred))
    bal = float(balanced_accuracy_score(true, pred))
    mf1 = float(f1_score(true, pred, labels=labels, average='macro', zero_division=0))
    p, r, f, s = precision_recall_fscore_support(true, pred, labels=labels, zero_division=0)
    per_class = {labels[i]: {'P': float(p[i]), 'R': float(r[i]),
                              'F1': float(f[i]), 'support': int(s[i])} for i in range(len(labels))}
    cm = confusion_matrix(true, pred, labels=labels).tolist()
    return {
        'accuracy': acc, 'balanced_acc': bal, 'macro_f1': mf1,
        'per_class': per_class, 'cm': cm, 'cm_labels': labels,
    }


def build_clip_anchors(model, tokenizer, device, labels=LABELS_5):
    """Prompt bank 평균 → 5-class anchor (L2-normalized)."""
    anchors = []
    with torch.no_grad():
        for lab in labels:
            tt = tokenizer(PROMPT_BANK[lab]).to(device)
            te = model.encode_text(tt)
            te = F.normalize(te, dim=-1).mean(dim=0, keepdim=True)
            te = F.normalize(te, dim=-1)
            anchors.append(te.cpu().numpy())
    return np.concatenate(anchors, axis=0)  # (5, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--out', type=str, default=str(OUT_DEFAULT))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST)

    print('[load] OpenCLIP ViT-B/32 (laion2b_s34b_b79k)...')
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=args.device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    anchors = build_clip_anchors(model, tokenizer, args.device, LABELS_5)  # (5, D)

    all_results = {'baselines': {}, 'note': 'v3 case 100 (V3-Ctrl) / 101 (V3-A) split 사용. 15 splits 평균.'}

    # 4종 베이스라인 각각에 대해 15-split 결과 집계
    baseline_names = ['majority', 'clip_zeroshot', 'clip_linprobe_og', 'clip_linprobe_og+syn']
    agg = {name: {'accuracy': [], 'balanced_acc': [], 'macro_f1': [], 'per_class_f1': {l: [] for l in LABELS_5}}
           for name in baseline_names}

    t0_total = time.time()
    for rep in sorted(df['repeat'].unique()):
        for fold in sorted(df['fold'].unique()):
            t0 = time.time()
            # 동일 (rep, fold)의 case 100·101 split만 사용
            sub_ctrl = df[(df['case'] == 100) & (df['repeat'] == rep) & (df['fold'] == fold)]
            sub_va = df[(df['case'] == 101) & (df['repeat'] == rep) & (df['fold'] == fold)]
            if len(sub_ctrl) == 0 or len(sub_va) == 0:
                continue
            ctrl_train = sub_ctrl[sub_ctrl.partition == 'train']
            test_df = sub_ctrl[sub_ctrl.partition == 'test']  # ctrl·va test 동일
            va_train = sub_va[sub_va.partition == 'train']

            # placeholder 경로(dry-run) 제외
            ctrl_train = ctrl_train[~ctrl_train['file_path'].str.startswith('[DRY-RUN-PLACEHOLDER]')]
            va_train = va_train[~va_train['file_path'].str.startswith('[DRY-RUN-PLACEHOLDER]')]

            test_paths = test_df['file_path'].tolist()
            test_labels = test_df['label'].tolist()
            ctrl_paths = ctrl_train['file_path'].tolist()
            ctrl_labels = ctrl_train['label'].tolist()
            va_paths = va_train['file_path'].tolist()
            va_labels = va_train['label'].tolist()

            test_emb = embed_batch(test_paths, model, preprocess, args.device)
            ctrl_emb = embed_batch(ctrl_paths, model, preprocess, args.device)
            va_emb = embed_batch(va_paths, model, preprocess, args.device)

            # 1) majority
            maj_label = Counter(ctrl_labels).most_common(1)[0][0]
            maj_pred = [maj_label] * len(test_labels)
            r1 = metrics_block(test_labels, maj_pred)

            # 2) clip_zeroshot — test_emb · anchors.T → argmax
            sim = test_emb @ anchors.T  # (Nt, 5)
            zs_pred = [LABELS_5[i] for i in sim.argmax(axis=1)]
            r2 = metrics_block(test_labels, zs_pred)

            # 3) clip_linprobe (OG only)
            r3 = None
            if len(set(ctrl_labels)) >= 2:
                clf3 = LogisticRegression(max_iter=2000, C=1.0, multi_class='auto')
                clf3.fit(ctrl_emb, ctrl_labels)
                lp_pred = clf3.predict(test_emb)
                r3 = metrics_block(test_labels, list(lp_pred))

            # 4) clip_linprobe (OG + Syn from V3-A)
            r4 = None
            if len(set(va_labels)) >= 2 and len(va_paths) > 0:
                clf4 = LogisticRegression(max_iter=2000, C=1.0, multi_class='auto')
                clf4.fit(va_emb, va_labels)
                lp2_pred = clf4.predict(test_emb)
                r4 = metrics_block(test_labels, list(lp2_pred))

            for name, r in zip(baseline_names, [r1, r2, r3, r4]):
                if r is None:
                    continue
                agg[name]['accuracy'].append(r['accuracy'])
                agg[name]['balanced_acc'].append(r['balanced_acc'])
                agg[name]['macro_f1'].append(r['macro_f1'])
                for lab in LABELS_5:
                    agg[name]['per_class_f1'][lab].append(r['per_class'][lab]['F1'])

            print(f'  r={rep} f={fold} '
                  f'maj_acc={r1["accuracy"]*100:.1f}% '
                  f'zs_acc={r2["accuracy"]*100:.1f}% '
                  f'lp_og_acc={(r3["accuracy"]*100 if r3 else float("nan")):.1f}% '
                  f'lp_ogsyn_acc={(r4["accuracy"]*100 if r4 else float("nan")):.1f}% '
                  f't={time.time()-t0:.0f}s')

    # 평균 정리
    summary = {}
    for name in baseline_names:
        a = agg[name]
        if not a['accuracy']:
            summary[name] = None
            continue
        summary[name] = {
            'n_splits': len(a['accuracy']),
            'accuracy_mean': float(np.mean(a['accuracy'])),
            'accuracy_std': float(np.std(a['accuracy'])),
            'balanced_acc_mean': float(np.mean(a['balanced_acc'])),
            'macro_f1_mean': float(np.mean(a['macro_f1'])),
            'macro_f1_std': float(np.std(a['macro_f1'])),
            'per_class_f1_mean': {l: float(np.mean(a['per_class_f1'][l])) for l in LABELS_5},
        }

    all_results['baselines'] = summary
    all_results['total_elapsed_sec'] = time.time() - t0_total
    with open(out_path, 'w') as fh:
        json.dump(all_results, fh, indent=2)
    print(f'\nSaved: {out_path}  (elapsed {all_results["total_elapsed_sec"]:.0f}s)')
    for name, s in summary.items():
        if s is None:
            print(f'  {name}: (skipped)')
            continue
        print(f'  {name}: acc={s["accuracy_mean"]*100:.1f}±{s["accuracy_std"]*100:.1f}% '
              f'mf1={s["macro_f1_mean"]*100:.1f}% (n={s["n_splits"]})')


if __name__ == '__main__':
    main()
