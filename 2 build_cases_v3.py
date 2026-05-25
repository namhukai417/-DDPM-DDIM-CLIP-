"""연구설계 v3 manifest 빌더.

핵심 차이 (v1/v2 ablation 대비):
1. 학습+test 모두 5라벨(A·C·V·C+V·others) 통합. test에 CV/others 포함.
2. StratifiedGroupKFold (num-aware + 클래스 비율 유지) + 클래스별 fold당 ≥1 num 보장 시드 재추출.
3. Syn 비율 분모 = OG_total 학습량 (v1은 OG_ACV였음). 90:10, 80:20, 70:30, 0:100 4 case.
4. Syn은 5라벨 모두 (A·C·V는 기존 풀, C·CV·others는 신규 풀 syn_filtered_v3/).
5. Syn에 source_num 메타데이터 부여 — OG num에서 파생되었으면 같은 partition으로 강제.
6. 총량 반올림은 ceil 적용 (산식 산출이 비정수일 때).
7. dry-run 모드 — 새 합성데이터가 아직 없어도 manifest 구조 검증 가능.

Usage:
    python build_cases_v3.py [--dry-run] [--seed-base 0]
"""
import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path('/home/kyungho2/kh/project_CLIP/2차연구')
OG_DIR = ROOT / 'data/og_by_class'
SYN_DIR_OLD = ROOT / 'data/syn_filtered'         # A·C·V (기존)
SYN_DIR_NEW = ROOT / 'data/syn_filtered_v3'      # C·CV·others (신규)
MANIFEST_OUT = ROOT / 'data/case_manifests/v3_cases.csv'
LABEL_DIRS = {'A':'A','C':'C','V':'V','C+V':'CV','others':'others'}
LABELS_5 = ['A','C','V','C+V','others']

CASES = {
    'V3-Ctrl': {'syn_ratio': 0.00},   # 0:100
    'V3-A':    {'syn_ratio': 0.90},   # 90:10
    'V3-B':    {'syn_ratio': 0.80},   # 80:20
    'V3-C':    {'syn_ratio': 0.70},   # 70:30
}
CASE_ORDER = ['V3-Ctrl', 'V3-A', 'V3-B', 'V3-C']
CASE_ID = {name: 100 + i for i, name in enumerate(CASE_ORDER)}  # case 100~103

N_FOLDS = 5
N_REPEATS = 3
MAX_SEED_RETRY = 200

NUM_REGEX = re.compile(r'^(\d+)\s*\(')


def extract_num(filename: str) -> int:
    m = NUM_REGEX.match(filename)
    if not m:
        raise ValueError(f'num 추출 실패: {filename}')
    return int(m.group(1))


def load_og() -> pd.DataFrame:
    """OG 이미지를 (num, label, file_path) 데이터프레임으로 적재."""
    rows = []
    for label, subdir in LABEL_DIRS.items():
        d = OG_DIR / subdir
        for f in sorted(d.glob('*.tif')):
            rows.append({'num': extract_num(f.name), 'label': label,
                         'label_suffix': subdir, 'file_path': str(f)})
    df = pd.DataFrame(rows)
    print(f'[OG] loaded {len(df)} images, {df["num"].nunique()} unique nums')
    print(df.groupby('label').size().to_string())
    # 데이터 누수 sanity: 한 num이 여러 라벨에 속하지 않는가?
    nlabel = df.groupby('num')['label'].nunique()
    bad = nlabel[nlabel > 1]
    assert len(bad) == 0, f'num이 여러 라벨에 속함: {bad.to_dict()}'
    return df


def load_syn_pool(dry_run: bool) -> dict:
    """Syn 풀을 라벨별로 적재. (file_path, source_num or None) tuple 리스트.

    SYN_DIR_OLD/{A,C,V}/ : 기존 풀. source_num 정보 없음(None).
    SYN_DIR_NEW/{C,CV,others}/ : 신규 풀. 파일명에 'src{num}_' 접두사가 있으면 source_num 추출.
                                  접두사 없으면 None (글로벌 모드).
    """
    pool = {lab: [] for lab in LABELS_5}
    # 5클래스 모두 SYN_DIR_NEW(syn_filtered_v3)에서 적재 — A·V도 동일 Quality-Plateau 필터 적용
    for lab in ['A', 'V']:
        d = SYN_DIR_NEW / lab
        if d.is_dir():
            files = sorted(d.glob('*.png')) + sorted(d.glob('*.tif'))
            for f in files:
                pool[lab].append({'file_path': str(f), 'source_num': None})
    # 신규 풀 — C(개선된 C v2), C+V, others
    for lab in ['C', 'C+V', 'others']:
        subdir = LABEL_DIRS[lab]
        d = SYN_DIR_NEW / subdir
        if d.is_dir():
            files = sorted(d.glob('*.png')) + sorted(d.glob('*.tif'))
            for f in files:
                m = re.match(r'^src(\d+)_', f.name)
                src_num = int(m.group(1)) if m else None
                pool[lab].append({'file_path': str(f), 'source_num': src_num})
    for lab in LABELS_5:
        print(f'[Syn pool] {lab}: {len(pool[lab])}')
    if not dry_run:
        # 90:10 비율 + 15 splits 추출 가능 여부 sanity
        for lab in LABELS_5:
            need = 50  # 대략 fold당 클래스 분배 상한 추정
            if len(pool[lab]) < need:
                print(f'[WARN] Syn pool[{lab}] 풀 작음 ({len(pool[lab])} < {need}) — '
                      f'90:10 case에서 추출 시 풀 부족 가능')
    return pool


def make_splits(df_og: pd.DataFrame, seed_base: int) -> list:
    """num-aware StratifiedGroupKFold × N_REPEATS, 모든 fold-class에 ≥1 num 보장.

    Returns: list of dicts {repeat, fold, train_idx, test_idx} (df_og row indices)
    """
    splits = []
    # num 단위로 (num, label) 쌍 추출
    num_label = df_og.groupby('num')['label'].first().reset_index()
    nums = num_label['num'].values
    labels = num_label['label'].values

    for rep in range(N_REPEATS):
        # 시드 재추출 루프
        seed = seed_base + rep * 1000
        attempts = 0
        ok = False
        while attempts < MAX_SEED_RETRY:
            skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed + attempts)
            try:
                fold_splits = list(skf.split(np.zeros(len(nums)), labels, groups=nums))
            except Exception as e:
                attempts += 1
                continue
            # 각 fold의 test에 모든 라벨이 ≥1장 들어 있는가?
            all_ok = True
            for tr_idx, te_idx in fold_splits:
                te_nums = nums[te_idx]
                te_labels = df_og[df_og['num'].isin(te_nums)]['label'].unique()
                if set(te_labels) != set(LABELS_5):
                    all_ok = False
                    break
            if all_ok:
                ok = True
                break
            attempts += 1
        assert ok, f'repeat {rep}: {MAX_SEED_RETRY}번 시도해도 모든 fold에 모든 라벨이 ≥1 안 들어감'
        print(f'[split] repeat {rep}: seed {seed + attempts} (attempts {attempts + 1}) — OK')

        for fold, (tr_idx, te_idx) in enumerate(fold_splits):
            tr_nums = set(nums[tr_idx])
            te_nums = set(nums[te_idx])
            assert len(tr_nums & te_nums) == 0, f'num leak at rep={rep} fold={fold}'
            tr_row_idx = df_og.index[df_og['num'].isin(tr_nums)].tolist()
            te_row_idx = df_og.index[df_og['num'].isin(te_nums)].tolist()
            splits.append({
                'repeat': rep, 'fold': fold,
                'train_idx': tr_row_idx, 'test_idx': te_row_idx,
                'train_nums': sorted(tr_nums), 'test_nums': sorted(te_nums),
            })
    return splits


def allocate_syn(og_train_count_by_label: dict, target_total_syn: int) -> dict:
    """OG_train 클래스 비율에 비례해서 Syn 클래스별 분배. ceil로 반올림."""
    total_og = sum(og_train_count_by_label.values())
    alloc = {}
    for lab in LABELS_5:
        frac = og_train_count_by_label[lab] / total_og if total_og else 0
        alloc[lab] = int(math.ceil(target_total_syn * frac))
    return alloc


def sample_syn(pool_list: list, n: int, test_nums: set, rng) -> list:
    """Syn 풀에서 n개 추출. source_num이 test_nums에 속하면 제외(간접 leakage 차단)."""
    eligible = [p for p in pool_list if p['source_num'] is None or p['source_num'] not in test_nums]
    if n == 0:
        return []
    if len(eligible) < n:
        # replace=True 폴백 (풀 부족 시) — 경고 출력
        print(f'  [WARN] eligible Syn ({len(eligible)}) < need ({n}); falling back to replace=True')
        idx = rng.integers(0, len(eligible), size=n)
    else:
        idx = rng.choice(len(eligible), size=n, replace=False)
    return [eligible[i] for i in idx]


def build_manifest(seed_base: int, dry_run: bool) -> pd.DataFrame:
    df_og = load_og()
    syn_pool = load_syn_pool(dry_run)
    splits = make_splits(df_og, seed_base)

    rows = []
    for sp in splits:
        rep, fold = sp['repeat'], sp['fold']
        train_og = df_og.loc[sp['train_idx']].copy()
        test_og  = df_og.loc[sp['test_idx']].copy()
        og_train_count = {lab: int((train_og['label'] == lab).sum()) for lab in LABELS_5}
        test_nums = set(sp['test_nums'])

        for case_name in CASE_ORDER:
            case_id = CASE_ID[case_name]
            syn_ratio = CASES[case_name]['syn_ratio']
            og_total = sum(og_train_count.values())
            target_syn = math.ceil(og_total * syn_ratio / (1 - syn_ratio)) if syn_ratio < 1 else 0
            alloc = allocate_syn(og_train_count, target_syn) if target_syn > 0 else {l: 0 for l in LABELS_5}

            # OG train rows
            for _, r in train_og.iterrows():
                rows.append({
                    'case': case_id, 'case_name': case_name,
                    'repeat': rep, 'fold': fold,
                    'partition': 'train', 'source': 'og',
                    'num': r['num'], 'sub': -1, 'source_num': r['num'],
                    'label': r['label'], 'label_suffix': r['label_suffix'],
                    'file_path': r['file_path'],
                })
            # Test OG rows (5라벨 모두 포함)
            for _, r in test_og.iterrows():
                rows.append({
                    'case': case_id, 'case_name': case_name,
                    'repeat': rep, 'fold': fold,
                    'partition': 'test', 'source': 'og',
                    'num': r['num'], 'sub': -1, 'source_num': r['num'],
                    'label': r['label'], 'label_suffix': r['label_suffix'],
                    'file_path': r['file_path'],
                })
            # Syn rows
            seed = case_id * 1000 + rep * 100 + fold
            for cls_offset, lab in enumerate(LABELS_5):
                n = alloc[lab]
                if n == 0:
                    continue
                rng = np.random.default_rng(seed * 10 + cls_offset)
                if not syn_pool[lab]:
                    if dry_run:
                        # dry-run: 가상 syn placeholder
                        for k in range(n):
                            rows.append({
                                'case': case_id, 'case_name': case_name,
                                'repeat': rep, 'fold': fold,
                                'partition': 'train', 'source': 'syn',
                                'num': -1, 'sub': -1, 'source_num': None,
                                'label': lab, 'label_suffix': LABEL_DIRS[lab],
                                'file_path': f'[DRY-RUN-PLACEHOLDER]/syn_{lab}_{k:04d}.png',
                            })
                        continue
                    else:
                        raise RuntimeError(f'Syn 풀 [{lab}] 비어있음 (case {case_name})')
                picked = sample_syn(syn_pool[lab], n, test_nums, rng)
                for s in picked:
                    rows.append({
                        'case': case_id, 'case_name': case_name,
                        'repeat': rep, 'fold': fold,
                        'partition': 'train', 'source': 'syn',
                        'num': -1, 'sub': -1, 'source_num': s['source_num'],
                        'label': lab, 'label_suffix': LABEL_DIRS[lab],
                        'file_path': s['file_path'],
                    })

    df = pd.DataFrame(rows)
    print(f'\n[manifest] total rows: {len(df)}')
    return df


def assertions(df: pd.DataFrame):
    """Manifest sanity 어서션."""
    # 1) 동일 num 중복 금지 (각 case·repeat·fold에서 train·test num 교집합 = ∅)
    for (c, r, f), grp in df.groupby(['case', 'repeat', 'fold']):
        tr_nums = set(grp[(grp['partition']=='train') & (grp['source']=='og')]['num'].unique())
        te_nums = set(grp[grp['partition']=='test']['num'].unique())
        assert not (tr_nums & te_nums), f'num leak: case{c} r{r} f{f}'
    # 2) Syn source_num이 test num과 겹치지 않음
    for (c, r, f), grp in df.groupby(['case', 'repeat', 'fold']):
        te_nums = set(grp[grp['partition']=='test']['num'].unique())
        syn = grp[grp['source']=='syn']
        bad = syn[syn['source_num'].notna() & syn['source_num'].isin(te_nums)]
        assert len(bad) == 0, f'Syn source_num leak: case{c} r{r} f{f}, n={len(bad)}'
    # 3) 동일 (repeat, fold)에서 모든 case의 OG test가 완전히 동일
    base_test = None
    for (r, f), grp in df.groupby(['repeat', 'fold']):
        per_case_test = {}
        for c, sub in grp.groupby('case'):
            te = sub[sub['partition']=='test']['file_path'].sort_values().tolist()
            per_case_test[c] = te
        first_case = sorted(per_case_test.keys())[0]
        for c in per_case_test:
            assert per_case_test[c] == per_case_test[first_case], \
                f'test set 불일치: r{r} f{f} case{c} vs case{first_case}'
    # 4) 모든 fold의 test에 5라벨 모두 ≥1
    for (c, r, f), grp in df.groupby(['case', 'repeat', 'fold']):
        te_labels = set(grp[grp['partition']=='test']['label'].unique())
        missing = set(LABELS_5) - te_labels
        if missing:
            print(f'  [WARN] case{c} r{r} f{f}: test에 라벨 누락: {missing}')
    print('[assertions] all passed ✓')


def report(df: pd.DataFrame):
    print('\n=== Case별 train/test 통계 ===')
    for case_name in CASE_ORDER:
        case_id = CASE_ID[case_name]
        sub = df[df['case'] == case_id]
        for r in range(N_REPEATS):
            for f in range(N_FOLDS):
                g = sub[(sub['repeat']==r) & (sub['fold']==f)]
                og_tr = ((g.partition=='train') & (g.source=='og')).sum()
                syn_tr = ((g.partition=='train') & (g.source=='syn')).sum()
                te = (g.partition=='test').sum()
                if r == 0 and f == 0:
                    print(f'{case_name} (case {case_id}) r{r}f{f}: OG train={og_tr}, Syn={syn_tr}, test={te}')
        # 클래스별 분배 (case 평균)
        tr_syn = sub[(sub.partition=='train') & (sub.source=='syn')]
        if len(tr_syn) > 0:
            print(f'  syn 분배: {dict(tr_syn.groupby("label").size())}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='Syn 신규풀 없이 placeholder로 manifest 구조 검증')
    ap.add_argument('--seed-base', type=int, default=20260511)
    args = ap.parse_args()
    df = build_manifest(args.seed_base, args.dry_run)
    assertions(df)
    report(df)
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_OUT, index=False)
    print(f'\nSaved: {MANIFEST_OUT}  ({len(df)} rows)')


if __name__ == '__main__':
    main()
