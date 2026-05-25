"""A·V 재선별 — filter_syn_quality.py와 동일 규칙(R1~R6).

입력:
  data/manifests/syn_quality_per_step.csv (cols: step, glcm_jsd, fft_l2, shape_wasserstein, class, ...)
  data/syn_by_class/{A,V}/step_<step>_<sub>.png

출력:
  data/syn_filtered_v3/{A,V}/  (각 672장)
  data/manifests/syn_filtered_v3_summary_av.md
  data/manifests/syn_filtered_v3_av.csv

규칙: R1 step ≥ 20%×max_step / R2 ≤ 80% / R3 composite mean min-max norm
      (GLCM·FFT·Shape) / R4 guard p60 / R5 top-42 / R6 16 imgs/step.
"""
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/kyungho2/kh/project_CLIP/2차연구')
QCSV = ROOT / 'data/manifests/syn_quality_per_step.csv'
OUT_DIR = ROOT / 'data/syn_filtered_v3'
MANIFEST_OUT = ROOT / 'data/manifests/syn_filtered_v3_av.csv'
SUMMARY_OUT = ROOT / 'data/manifests/syn_filtered_v3_summary_av.md'

METRICS = ['glcm_jsd', 'fft_l2', 'shape_wasserstein']
N_TARGET_STEPS = 42
GUARD_PCT = 0.60
LO_FRAC = 0.20
HI_FRAC = 0.80
CLASSES = ['A', 'V']


def select_steps(sub: pd.DataFrame):
    max_step = int(sub['step'].max())
    lo = LO_FRAC * max_step
    hi = HI_FRAC * max_step
    win = sub[(sub['step'] >= lo) & (sub['step'] <= hi)].copy().reset_index(drop=True)

    norm, guard_thr, mm = {}, {}, {}
    for m in METRICS:
        v = win[m].astype(float).values
        vmin, vmax = v.min(), v.max()
        rng = vmax - vmin if vmax > vmin else 1.0
        norm[m] = (v - vmin) / rng
        guard_thr[m] = float(np.quantile(v, GUARD_PCT))
        mm[m] = (float(vmin), float(vmax))

    composite = np.mean([norm[m] for m in METRICS], axis=0)
    win['composite'] = composite

    pass_mask = np.ones(len(win), dtype=bool)
    for m in METRICS:
        pass_mask &= (win[m].values <= guard_thr[m])
    passed = win[pass_mask].sort_values('composite').reset_index(drop=True)

    if len(passed) >= N_TARGET_STEPS:
        chosen = passed.head(N_TARGET_STEPS)
        rule = f'composite top-{N_TARGET_STEPS} (guard ≤ p{int(GUARD_PCT*100)})'
    elif len(passed) > 0:
        chosen = win.sort_values('composite').head(N_TARGET_STEPS)
        rule = f'guard relaxed → composite top-{N_TARGET_STEPS} (only {len(passed)} passed guard)'
    else:
        chosen = win.sort_values('composite').head(N_TARGET_STEPS)
        rule = 'fallback: composite top-N without guard'

    kept = sorted(chosen['step'].astype(int).tolist())
    stats = {
        'max_step': max_step,
        'window': (int(lo), int(hi)),
        'window_n_steps': len(win),
        'per_metric_best': mm,
        'guard_thr': guard_thr,
        'n_pass_guard': int(pass_mask.sum()),
        'rule_used': rule,
        'n_chosen': len(kept),
        'composite_range_chosen': (float(chosen['composite'].min()),
                                   float(chosen['composite'].max())),
        'kept_step_range': (min(kept), max(kept)),
    }
    return kept, stats


def main():
    df = pd.read_csv(QCSV)
    summary = ['# A·V 재선별 (Quality-Plateau v3 규칙 동일 적용)', '']
    summary += [f'- N_TARGET_STEPS = {N_TARGET_STEPS}',
                f'- GUARD_PERCENTILE = p{int(GUARD_PCT*100)}',
                f'- WINDOW = [{int(LO_FRAC*100)}%, {int(HI_FRAC*100)}%] of max_step',
                f'- COMPOSITE = mean of min-max normalized (GLCM JSD, FFT L2, Shape Wass)',
                '']
    mani_rows = []
    for cls in CLASSES:
        sub = df[df['class'] == cls].copy()
        kept, stats = select_steps(sub)
        src_dir = ROOT / 'data/syn_by_class' / cls
        out_dir = OUT_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        # purge old
        for old in out_dir.glob('*.png'):
            old.unlink()
        for s in kept:
            prefix = f'step_{s:06d}_'
            for f in sorted(src_dir.glob(f'{prefix}*.png')):
                dst = out_dir / f.name
                shutil.copy2(f, dst)
                mani_rows.append({'class': cls, 'step': s, 'filename': f.name,
                                  'src': str(f), 'dst': str(dst),
                                  'rule': stats['rule_used']})
        bmin = stats['per_metric_best']
        gthr = stats['guard_thr']
        msg = (
            f'## {cls}\n'
            f'- max_step: {stats["max_step"]:,}, window: {stats["window"][0]:,}~{stats["window"][1]:,} '
            f'({stats["window_n_steps"]} steps)\n'
            f'- window min/max (GLCM / FFT / Shape): '
            f'{bmin["glcm_jsd"][0]:.4f}~{bmin["glcm_jsd"][1]:.4f} / '
            f'{bmin["fft_l2"][0]:.4f}~{bmin["fft_l2"][1]:.4f} / '
            f'{bmin["shape_wasserstein"][0]:.4f}~{bmin["shape_wasserstein"][1]:.4f}\n'
            f'- guard p{int(GUARD_PCT*100)} (GLCM / FFT / Shape): '
            f'{gthr["glcm_jsd"]:.4f} / {gthr["fft_l2"]:.4f} / {gthr["shape_wasserstein"]:.4f}\n'
            f'- passed guard (all 3 metrics ≤ thr): {stats["n_pass_guard"]} steps\n'
            f'- rule used: **{stats["rule_used"]}**\n'
            f'- composite range chosen: {stats["composite_range_chosen"][0]:.3f} ~ {stats["composite_range_chosen"][1]:.3f}\n'
            f'- final kept: **{len(kept)} steps × 16 imgs = {len(kept)*16} images**\n'
            f'- kept step range: {stats["kept_step_range"][0]:,} ~ {stats["kept_step_range"][1]:,}\n'
        )
        summary.append(msg)
        print(msg)
    mdf = pd.DataFrame(mani_rows)
    mdf.to_csv(MANIFEST_OUT, index=False)
    SUMMARY_OUT.write_text('\n'.join(summary))
    print(f'[saved] {MANIFEST_OUT} ({len(mdf)} rows)')
    print(f'[saved] {SUMMARY_OUT}')


if __name__ == '__main__':
    main()
