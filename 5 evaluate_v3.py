"""V3 단일 (case, repeat, fold) fine-tune + 5-class 평가.

clip_finetune.py(v1/v2)와 다른 점:
1. manifest = v3_cases.csv (case 100~103, V3-Ctrl/A/B/C)
2. 학습·평가 모두 5라벨(A·C·V·C+V·others). test 자체에 CV/others 포함
3. 평가 지표: 5-class accuracy / balanced acc / macro-F1 / per-class F1 / 5x5 CM
4. 결과 파일은 results_v3/

Usage:
  python evaluate_v3.py --case 100 --repeat 0 --fold 0 --device cuda:0
"""
import os
import json
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix,
)

ROOT = Path('/home/kyungho2/kh/project_CLIP/2차연구')
MANIFEST = ROOT / 'data/case_manifests/v3_cases.csv'
RESULTS = ROOT / 'results_v3'
RESULTS.mkdir(parents=True, exist_ok=True)

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


class SEMDataset(Dataset):
    def __init__(self, df, preprocess):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row['file_path']).convert('RGB')
        x = self.preprocess(img)
        return x, row['label']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics_5class(true, pred, labels=LABELS_5):
    acc = accuracy_score(true, pred)
    bal = balanced_accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, labels=labels, average='macro', zero_division=0)
    p, r, f, s = precision_recall_fscore_support(true, pred, labels=labels, zero_division=0)
    per_class = {labels[i]: {'P': float(p[i]), 'R': float(r[i]),
                              'F1': float(f[i]), 'support': int(s[i])} for i in range(len(labels))}
    cm = confusion_matrix(true, pred, labels=labels).tolist()
    return {
        'accuracy': float(acc),
        'balanced_acc': float(bal),
        'macro_f1': float(macro_f1),
        'per_class': per_class,
        'cm': cm,
        'cm_labels': labels,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=int, required=True, help='100, 101, 102, 103')
    ap.add_argument('--repeat', type=int, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device
    t0 = time.time()

    # 1) 데이터 로드
    df = pd.read_csv(MANIFEST)
    sub = df[(df['case'] == args.case) & (df['repeat'] == args.repeat) & (df['fold'] == args.fold)]
    if len(sub) == 0:
        raise SystemExit(f'manifest에 (case={args.case}, repeat={args.repeat}, fold={args.fold}) 항목 없음')
    train_df = sub[sub.partition == 'train'].reset_index(drop=True)
    test_df = sub[sub.partition == 'test'].reset_index(drop=True)
    case_name = sub['case_name'].iloc[0]

    # 학습 라벨은 train_df에 등장한 라벨로 결정 (v3는 항상 5라벨)
    train_labels_local = [l for l in LABELS_5 if (train_df['label'] == l).any()]
    if len(train_labels_local) < len(LABELS_5):
        # V3-Ctrl(0:100) + 특정 fold에서 OG만으로 5라벨이 다 안 채워질 수도 있으나 manifest 단계에서 ≥1 num 보장됨
        print(f'[WARN] train 라벨 부족: {set(LABELS_5) - set(train_labels_local)}')

    # 2) 모델
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.train()

    # 3) Prompt 토큰
    train_prompt_tokens = {lab: tokenizer(PROMPT_BANK[lab]).to(device) for lab in train_labels_local}

    # 4) DataLoader
    train_ds = SEMDataset(train_df, preprocess)
    test_ds = SEMDataset(test_df, preprocess)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # 5) Optimizer + scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = max(1, args.epochs * len(train_loader))

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        return max(0.05, (total_steps - step) / max(1, total_steps - args.warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # 6) 학습 루프 — image-text InfoNCE (clip_finetune.py와 동일)
    rng = np.random.default_rng(args.seed)
    losses = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        nbat = 0
        for imgs, labs in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            text_tokens = []
            for lab in labs:
                idx = rng.integers(5)
                text_tokens.append(train_prompt_tokens[lab][idx])
            text_tokens = torch.stack(text_tokens, dim=0)

            img_emb = model.encode_image(imgs)
            txt_emb = model.encode_text(text_tokens)
            img_emb = F.normalize(img_emb, dim=-1)
            txt_emb = F.normalize(txt_emb, dim=-1)
            logit_scale = model.logit_scale.exp().clamp(max=100)
            logits_i2t = logit_scale * img_emb @ txt_emb.T
            logits_t2i = logits_i2t.T
            target = torch.arange(len(imgs), device=device)
            loss = (F.cross_entropy(logits_i2t, target) + F.cross_entropy(logits_t2i, target)) / 2

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, np.log(100))
            epoch_loss += loss.item()
            nbat += 1
        losses.append(epoch_loss / max(1, nbat))

    # 7) 추론 — 5-class anchor 생성 → argmax
    model.eval()
    with torch.no_grad():
        anchors = {}
        for lab in train_labels_local:
            tt = train_prompt_tokens[lab]
            te = model.encode_text(tt)
            te = F.normalize(te, dim=-1)
            te = te.mean(dim=0, keepdim=True)
            te = F.normalize(te, dim=-1)
            anchors[lab] = te
        anchor_mat = torch.cat([anchors[l] for l in train_labels_local], dim=0)

        all_preds = []
        all_true = []
        all_probs = []
        for imgs, labs in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            ie = model.encode_image(imgs)
            ie = F.normalize(ie, dim=-1)
            sim = ie @ anchor_mat.T
            prob = F.softmax(sim * model.logit_scale.exp().clamp(max=100), dim=-1)
            pred = sim.argmax(dim=-1).cpu().numpy()
            all_preds.extend([train_labels_local[i] for i in pred])
            all_true.extend(list(labs))
            all_probs.append(prob.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # 8) Metric
    m = metrics_5class(all_true, all_preds, labels=LABELS_5)

    elapsed = time.time() - t0
    result = {
        'case': args.case,
        'case_name': case_name,
        'repeat': args.repeat,
        'fold': args.fold,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'train_n': len(train_df),
        'test_n': len(test_df),
        'train_labels_used': train_labels_local,
        'elapsed_sec': elapsed,
        'final_loss': losses[-1] if losses else None,
        'metrics': m,
        'preds': all_preds,
        'true': all_true,
        'probs': all_probs.tolist(),
        'prob_labels': train_labels_local,
        'test_paths': test_df['file_path'].tolist(),
    }
    out_path = RESULTS / f'case{args.case}_r{args.repeat}_f{args.fold}.json'
    with open(out_path, 'w') as fh:
        json.dump(result, fh)

    print(
        f'[{case_name} case={args.case} r={args.repeat} f={args.fold}] '
        f'acc={m["accuracy"]*100:.1f}% bal={m["balanced_acc"]*100:.1f}% '
        f'mf1={m["macro_f1"]*100:.1f}% '
        f't={elapsed:.0f}s loss={(losses[-1] if losses else float("nan")):.3f}'
    )


if __name__ == '__main__':
    main()
