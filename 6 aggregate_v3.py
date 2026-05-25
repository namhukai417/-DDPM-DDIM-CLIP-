"""V3 결과 60개 JSON → 통계 요약·표·CSV.

산출:
  - results_v3/aggregate_summary.csv (case별 mean ± std)
  - results_v3/aggregate_per_class.csv (case×label별 P/R/F1)
  - results_v3/aggregate_confusion_v3-A.csv 등 (case별 합산 confusion matrix)
  - results_v3/aggregate_pairwise.csv (case 쌍별 paired t / Wilcoxon)
  - results_v3/aggregate_summary.md (markdown report)

Usage:
  python aggregate_v3.py
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path('/home/kyungho2/kh/project_CLIP/2차연구')
RESULTS = ROOT / 'results_v3'
CASES = [100, 101, 102, 103]
CASE_NAME = {100: 'V3-Ctrl', 101: 'V3-A', 102: 'V3-B', 103: 'V3-C'}
LABELS_5 = ['A', 'C', 'V', 'C+V', 'others']
REPEATS = [0, 1, 2]
FOLDS = [0, 1, 2, 3, 4]


def load_all():
    rows = []
    missing = []
    for c in CASES:
        for r in REPEATS:
            for f in FOLDS:
                p = RESULTS / f'case{c}_r{r}_f{f}.json'
                if not p.exists():
                    missing.append((c, r, f))
                    continue
                with open(p) as fh:
                    rj = json.load(fh)
                rows.append({
                    'case': c, 'case_name': CASE_NAME[c],
                    'repeat': r, 'fold': f,
                    'accuracy': rj['metrics']['accuracy'],
                    'balanced_acc': rj['metrics']['balanced_acc'],
                    'macro_f1': rj['metrics']['macro_f1'],
                    'per_class': rj['metrics']['per_class'],
                    'cm': np.array(rj['metrics']['cm']),
                    'final_loss': rj.get('final_loss'),
                    'train_n': rj.get('train_n'),
                    'test_n': rj.get('test_n'),
                    'elapsed_sec': rj.get('elapsed_sec'),
                })
    return rows, missing


def _safe_mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else float('nan')


def summary_table(rows):
    """case별 mean ± std."""
    out = []
    for c in CASES:
        sub = [r for r in rows if r['case'] == c]
        if not sub:
            out.append({'case': c, 'case_name': CASE_NAME[c], 'n': 0})
            continue
        d = {
            'case': c,
            'case_name': CASE_NAME[c],
            'n': len(sub),
            'accuracy_mean': np.mean([s['accuracy'] for s in sub]),
            'accuracy_std': np.std([s['accuracy'] for s in sub], ddof=1) if len(sub) > 1 else 0.0,
            'balanced_acc_mean': np.mean([s['balanced_acc'] for s in sub]),
            'balanced_acc_std': np.std([s['balanced_acc'] for s in sub], ddof=1) if len(sub) > 1 else 0.0,
            'macro_f1_mean': np.mean([s['macro_f1'] for s in sub]),
            'macro_f1_std': np.std([s['macro_f1'] for s in sub], ddof=1) if len(sub) > 1 else 0.0,
            'final_loss_mean': _safe_mean([s['final_loss'] for s in sub]),
            'train_n_mean': _safe_mean([s['train_n'] for s in sub]),
            'test_n_mean': _safe_mean([s['test_n'] for s in sub]),
            'elapsed_sec_mean': _safe_mean([s['elapsed_sec'] for s in sub]),
        }
        out.append(d)
    return pd.DataFrame(out)


def per_class_table(rows):
    """case × label별 P/R/F1 평균."""
    out = []
    for c in CASES:
        sub = [r for r in rows if r['case'] == c]
        if not sub:
            continue
        for lab in LABELS_5:
            ps = [s['per_class'][lab]['P'] for s in sub if lab in s['per_class']]
            rs = [s['per_class'][lab]['R'] for s in sub if lab in s['per_class']]
            f1s = [s['per_class'][lab]['F1'] for s in sub if lab in s['per_class']]
            sup = [s['per_class'][lab]['support'] for s in sub if lab in s['per_class']]
            out.append({
                'case': c, 'case_name': CASE_NAME[c], 'label': lab,
                'P_mean': np.mean(ps), 'P_std': np.std(ps, ddof=1) if len(ps) > 1 else 0.0,
                'R_mean': np.mean(rs), 'R_std': np.std(rs, ddof=1) if len(rs) > 1 else 0.0,
                'F1_mean': np.mean(f1s), 'F1_std': np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0,
                'support_mean': np.mean(sup),
            })
    return pd.DataFrame(out)


def case_cm(rows):
    """case별 confusion matrix 합산."""
    out = {}
    for c in CASES:
        sub = [r for r in rows if r['case'] == c]
        if not sub:
            continue
        cm = np.zeros_like(sub[0]['cm'])
        for s in sub:
            cm = cm + s['cm']
        out[c] = cm
    return out


def pairwise_tests(rows):
    """case 쌍별 paired t / Wilcoxon for accuracy·balanced_acc·macro_f1.

    페어링은 같은 (repeat, fold) — 결측 시 해당 페어 drop.
    """
    out = []
    by_case = {c: {(s['repeat'], s['fold']): s for s in rows if s['case'] == c} for c in CASES}
    metrics = ['accuracy', 'balanced_acc', 'macro_f1']
    for c1 in CASES:
        for c2 in CASES:
            if c1 >= c2:
                continue
            common = sorted(set(by_case[c1].keys()) & set(by_case[c2].keys()))
            if len(common) < 3:
                continue
            for m in metrics:
                a = np.array([by_case[c1][k][m] for k in common])
                b = np.array([by_case[c2][k][m] for k in common])
                d = b - a
                # paired t
                if np.allclose(d, 0):
                    t_p = 1.0
                    t_stat = 0.0
                else:
                    t_stat, t_p = stats.ttest_rel(b, a)
                # Wilcoxon (zero-method='wilcox' default)
                try:
                    w_stat, w_p = stats.wilcoxon(b, a, zero_method='wilcox')
                except ValueError:
                    w_stat, w_p = float('nan'), float('nan')
                out.append({
                    'compare': f'{CASE_NAME[c2]} - {CASE_NAME[c1]}',
                    'metric': m,
                    'n_pairs': len(common),
                    'mean_diff': float(np.mean(d)),
                    'std_diff': float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                    't_stat': float(t_stat), 't_p': float(t_p),
                    'w_stat': float(w_stat) if not np.isnan(w_stat) else float('nan'),
                    'w_p': float(w_p) if not np.isnan(w_p) else float('nan'),
                })
    return pd.DataFrame(out)


def md_report(summ, per_cls, cm_dict, pair, missing):
    lines = ['# V3 실험 결과 집계 보고', '']
    lines.append(f'- 완료 split: {int(summ["n"].sum())} / 60')
    lines.append(f'- 결측 split: {len(missing)}')
    if missing:
        lines.append(f'  - missing: {missing[:10]}{"..." if len(missing) > 10 else ""}')
    lines.append('')
    lines.append('## case별 5-class 평가 (mean ± std)')
    cols = ['case_name', 'n', 'accuracy_mean', 'accuracy_std',
            'balanced_acc_mean', 'balanced_acc_std',
            'macro_f1_mean', 'macro_f1_std', 'elapsed_sec_mean']
    sub = summ[[c for c in cols if c in summ.columns]].copy()
    for c in sub.columns:
        if sub[c].dtype == float:
            sub[c] = sub[c].round(4)
    lines.append(sub.to_markdown(index=False))
    lines.append('')
    lines.append('## case × label per-class F1 (mean)')
    pivot = per_cls.pivot(index='case_name', columns='label', values='F1_mean').round(4)
    pivot = pivot.reindex(columns=LABELS_5)
    lines.append(pivot.to_markdown())
    lines.append('')
    lines.append('## 페어드 검정 (V3-A/B/C vs V3-Ctrl 위주)')
    pair_show = pair[pair['compare'].str.contains('V3-Ctrl')].copy()
    if not pair_show.empty:
        for c in pair_show.columns:
            if pair_show[c].dtype == float:
                pair_show[c] = pair_show[c].round(4)
        lines.append(pair_show.to_markdown(index=False))
    lines.append('')
    for c, cm in cm_dict.items():
        lines.append(f'## Confusion Matrix — {CASE_NAME[c]} (rows=true, cols=pred)')
        cm_df = pd.DataFrame(cm, index=LABELS_5, columns=LABELS_5)
        lines.append(cm_df.to_markdown())
        lines.append('')
    return '\n'.join(lines)


def main():
    rows, missing = load_all()
    print(f'loaded {len(rows)}, missing {len(missing)}')
    if not rows:
        print('결과 없음. evaluate_v3 먼저 실행하세요.')
        return

    summ = summary_table(rows)
    per_cls = per_class_table(rows)
    cm_dict = case_cm(rows)
    pair = pairwise_tests(rows)

    summ.to_csv(RESULTS / 'aggregate_summary.csv', index=False)
    per_cls.to_csv(RESULTS / 'aggregate_per_class.csv', index=False)
    pair.to_csv(RESULTS / 'aggregate_pairwise.csv', index=False)
    for c, cm in cm_dict.items():
        pd.DataFrame(cm, index=LABELS_5, columns=LABELS_5).to_csv(
            RESULTS / f'aggregate_cm_{CASE_NAME[c]}.csv')

    md = md_report(summ, per_cls, cm_dict, pair, missing)
    (RESULTS / 'aggregate_summary.md').write_text(md)
    print(f'wrote: {RESULTS / "aggregate_summary.csv"}')
    print(f'wrote: {RESULTS / "aggregate_per_class.csv"}')
    print(f'wrote: {RESULTS / "aggregate_pairwise.csv"}')
    print(f'wrote: {RESULTS / "aggregate_summary.md"}')


if __name__ == '__main__':
    main()
