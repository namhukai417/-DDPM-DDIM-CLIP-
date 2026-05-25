"""Quality-plateau 기반 syn 이미지 통일 필터.

lightweight_eval_all_models.md의 F1~F4 발견을 직접 필터 규칙으로 변환:
  F1 (data ↔ quality 직선) → 클래스별 합격 step 수가 달라질 수 있음
  F2 (Loss best ≠ Quality best) → CLIP-anchor 의미 필터 대신 측정 시각 품질만 사용
  F3 (late mode collapse) → step > 80% × max_step 제거
  F4 (plateau 시점 차이) → 절대 step 범위 하드코딩 금지, 데이터 driven

규칙 (모든 클래스에 동일 적용):
  R1. 카오스 초반 제거: step < 0.20 × max_step
  R2. Late collapse 제거: step > 0.80 × max_step
  R3. Composite quality score:
        각 지표(GLCM JSD / FFT L2 / Shape Wass)를 window 안에서
        (x - x_min) / (x_max - x_min) 정규화 후 단순 평균.
        값이 작을수록 우수.
  R4. Composite 점수 하위(=우수) N_TARGET_STEPS 선택.
        하한 가드: 어떤 지표든 ≤ 그 지표의 60th-percentile (즉 worse-half 자동 제외).
  R5. N_TARGET_STEPS = 42 (1차연구 컨벤션, 42 × 16 = 672 imgs).
  R6. 각 step의 16장 모두 보존.

Usage:
    python filter_syn_quality.py [--dry-run] [--n-steps 42] [--guard-pct 0.60]
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/kyungho2/kh/project_CLIP/2차연구')
OUT_DIR = ROOT / 'data/syn_filtered_v3'
MANIFEST_OUT = ROOT / 'data/manifests/syn_filtered_v3.csv'
SUMMARY_OUT = ROOT / 'data/manifests/syn_filtered_v3_summary.md'

# (src_dir_name, out_label, max_step_hint) — 입력 경로와 출력 라벨
CLASSES = [
    ('C v2',  'C',      500000),
    ('C+V',   'CV',     300000),
    ('others','others', 300000),
]

METRICS = ['glcm_jsd', 'fft_l2', 'shape_wasserstein']
N_TARGET_STEPS = 42
GUARD_PCT = 0.60  # 어떤 지표든 worse-half(>60th percentile) step은 제외
LO_FRAC = 0.20
HI_FRAC = 0.80


def detect_quality_steps(csv_path: Path, max_step_hint: int, n_target: int, guard_pct: float,
                         lo_frac: float, hi_frac: float) -> tuple:
    """Composite quality score 기반 우수 step 선정.

    1) window = [lo_frac, hi_frac] × max_step
    2) 각 지표 window 안에서 min-max 정규화 (작을수록 우수, 0이 best)
    3) 어떤 지표든 그 지표의 guard_pct-percentile보다 worse한 step은 탈락
    4) 통과한 step 중 composite (평균) 점수 작은 순으로 n_target 선택

    Returns:
        (kept_steps: sorted list[int], stats: dict)
    """
    df = pd.read_csv(csv_path)
    max_step = max(df['step'].max(), max_step_hint)
    lo, hi = lo_frac * max_step, hi_frac * max_step

    window = df[(df['step'] >= lo) & (df['step'] <= hi)].copy().reset_index(drop=True)

    per_metric_best = {}
    norm = {}
    guard_thr = {}
    for m in METRICS:
        v = window[m].astype(float).values
        vmin, vmax = v.min(), v.max()
        rng = vmax - vmin if vmax > vmin else 1.0
        norm[m] = (v - vmin) / rng                      # 0(best) ~ 1(worst)
        guard_thr[m] = float(np.quantile(v, guard_pct))
        per_metric_best[m] = (float(vmin), float(vmax))

    composite = np.mean([norm[m] for m in METRICS], axis=0)
    window['composite'] = composite

    # guard 통과 마스크
    pass_mask = np.ones(len(window), dtype=bool)
    for m in METRICS:
        pass_mask &= (window[m].values <= guard_thr[m])

    passed = window[pass_mask].sort_values('composite').reset_index(drop=True)
    n_pass = len(passed)

    if n_pass >= n_target:
        chosen = passed.head(n_target)
        rule = f'composite top-{n_target} (guard ≤ p{int(guard_pct*100)})'
    elif n_pass > 0:
        # guard 통과가 부족: guard 완화하여 composite top-n_target
        relax = window.sort_values('composite').head(n_target)
        chosen = relax
        rule = f'guard relaxed → composite top-{n_target} (only {n_pass} passed guard)'
    else:
        chosen = window.sort_values('composite').head(n_target)
        rule = 'fallback: composite top-N without guard'

    kept_steps = sorted(chosen['step'].astype(int).tolist())

    stats = {
        'max_step': int(max_step),
        'window': (int(lo), int(hi)),
        'window_n_steps': len(window),
        'per_metric_best': per_metric_best,
        'guard_thr': guard_thr,
        'n_pass_guard': n_pass,
        'rule_used': rule,
        'n_chosen': len(kept_steps),
        'composite_range_chosen': (
            float(chosen['composite'].min()),
            float(chosen['composite'].max()),
        ),
    }
    return kept_steps, stats


def gather_images(src_dir: Path, kept_steps: list) -> list:
    """선정된 step의 모든 PNG를 (step, filename, path)로 반환."""
    rows = []
    for s in kept_steps:
        prefix = f'step_{s:06d}_'
        for f in sorted(src_dir.glob(f'{prefix}*.png')):
            rows.append({'step': s, 'filename': f.name, 'src': str(f)})
    return rows


def main(dry_run: bool, n_steps: int, guard_pct: float):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    summary_lines = ['# syn_filtered_v3 quality-plateau 필터 결과', '']
    summary_lines += [f'- N_TARGET_STEPS = {n_steps}',
                      f'- GUARD_PERCENTILE = p{int(guard_pct*100)} (any metric worse than this → drop)',
                      f'- WINDOW = [{int(LO_FRAC*100)}%, {int(HI_FRAC*100)}%] of max_step',
                      f'- COMPOSITE = mean of min-max normalized (GLCM JSD, FFT L2, Shape Wass)',
                      '']

    for src_name, out_label, max_step_hint in CLASSES:
        src_dir = ROOT / 'data' / src_name / 'individual'
        csv_path = ROOT / 'data' / src_name / 'lightweight_eval' / 'lightweight_eval.csv'
        if not csv_path.exists():
            print(f'[SKIP] {src_name}: lightweight_eval.csv 없음')
            continue

        kept, stats = detect_quality_steps(csv_path, max_step_hint, n_steps, guard_pct, LO_FRAC, HI_FRAC)
        imgs = gather_images(src_dir, kept)

        # --- 출력 ---
        out_class_dir = OUT_DIR / out_label
        if not dry_run:
            out_class_dir.mkdir(parents=True, exist_ok=True)
            # 기존 파일 제거(부분 산출물 방지)
            for old in out_class_dir.glob('*.png'):
                old.unlink()
            for r in imgs:
                dst = out_class_dir / r['filename']
                shutil.copy2(r['src'], dst)
                manifest_rows.append({
                    'class': out_label, 'step': r['step'],
                    'filename': r['filename'],
                    'src': r['src'], 'dst': str(dst),
                    'rule': stats['rule_used'],
                })
        else:
            for r in imgs:
                manifest_rows.append({
                    'class': out_label, 'step': r['step'],
                    'filename': r['filename'],
                    'src': r['src'], 'dst': '[DRY-RUN]',
                    'rule': stats['rule_used'],
                })

        # --- 요약 ---
        bmin = stats['per_metric_best']
        gthr = stats['guard_thr']
        msg = (
            f'## {src_name} → {out_label}\n'
            f'- max_step: {stats["max_step"]:,}, window: {stats["window"][0]:,}~{stats["window"][1]:,} '
            f'({stats["window_n_steps"]} steps)\n'
            f'- window min/max (GLCM / FFT / Shape): '
            f'{bmin["glcm_jsd"][0]:.4f}~{bmin["glcm_jsd"][1]:.4f} / '
            f'{bmin["fft_l2"][0]:.4f}~{bmin["fft_l2"][1]:.4f} / '
            f'{bmin["shape_wasserstein"][0]:.4f}~{bmin["shape_wasserstein"][1]:.4f}\n'
            f'- guard p{int(guard_pct*100)} (GLCM / FFT / Shape): '
            f'{gthr["glcm_jsd"]:.4f} / {gthr["fft_l2"]:.4f} / {gthr["shape_wasserstein"]:.4f}\n'
            f'- passed guard (all 3 metrics ≤ thr): {stats["n_pass_guard"]} steps\n'
            f'- rule used: **{stats["rule_used"]}**\n'
            f'- composite range chosen: {stats["composite_range_chosen"][0]:.3f} ~ {stats["composite_range_chosen"][1]:.3f}\n'
            f'- final kept: **{len(kept)} steps × 16 imgs = {len(imgs)} images**\n'
            f'- kept step range: {min(kept):,} ~ {max(kept):,}\n'
        )
        summary_lines.append(msg)
        print(msg)

    df = pd.DataFrame(manifest_rows)
    if not dry_run:
        df.to_csv(MANIFEST_OUT, index=False)
        with open(SUMMARY_OUT, 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f'\n[saved] {MANIFEST_OUT}  ({len(df)} rows)')
        print(f'[saved] {SUMMARY_OUT}')
    else:
        print(f'\n[DRY-RUN] would write {len(df)} rows to {MANIFEST_OUT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--n-steps', type=int, default=N_TARGET_STEPS)
    ap.add_argument('--guard-pct', type=float, default=GUARD_PCT)
    args = ap.parse_args()
    main(args.dry_run, args.n_steps, args.guard_pct)
