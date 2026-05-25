#!/usr/bin/env bash
# V3 파이프라인 일괄 실행.
#
# 사전 조건
#   - /home/kyungho2/kh/project_CLIP/2차연구/data/og_by_class/{A,C,V,CV,others}/ 존재
#   - syn_filtered/ (기존 A·C·V 풀) 존재
#   - syn_filtered_v3/{C,CV,others}/ 신규 풀 준비 완료 (--dry-run으로는 placeholder 가능)
#   - myenv 또는 .venv 활성화, open_clip / torch / sklearn 설치
#
# 단계
#   1) v3 manifest 빌드 (dry-run 또는 real)
#   2) domain_gap 산출 (OG↔Syn 격차)
#   3) baselines 산출 (majority / zero-shot / linear probe)
#   4) 60-split fine-tune+평가 (run_v3_safe.py)
#   5) aggregate (별도 스크립트 — TODO)
set -euo pipefail

ROOT="/home/kyungho2/kh/project_CLIP/2차연구"
CODE="${ROOT}/code"
LOGS="${ROOT}/logs_v3"
mkdir -p "${LOGS}"

# 가상환경 자동 활성화 (있을 때만)
if [[ -f "/home/kyungho2/myenv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source /home/kyungho2/myenv/bin/activate
elif [[ -f "/home/kyungho2/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source /home/kyungho2/.venv/bin/activate
fi

MODE="${1:-real}"   # real | dry-run | manifest-only | baselines-only | finetune-only
echo "[run_v3] mode=${MODE}"

case "${MODE}" in
    dry-run)
        echo "[1/2] manifest (dry-run)"
        python "${CODE}/build_cases_v3.py" --dry-run 2>&1 | tee "${LOGS}/build_cases.log"
        echo "[2/2] (dry-run에서는 fine-tune·baseline·domain_gap 생략)"
        ;;

    manifest-only)
        echo "[1/1] manifest (real)"
        python "${CODE}/build_cases_v3.py" 2>&1 | tee "${LOGS}/build_cases.log"
        ;;

    baselines-only)
        echo "[1/2] domain_gap"
        python "${CODE}/domain_gap.py" --device cuda:0 2>&1 | tee "${LOGS}/domain_gap.log"
        echo "[2/2] baselines"
        python "${CODE}/baseline_v3.py" --device cuda:0 2>&1 | tee "${LOGS}/baselines.log"
        ;;

    finetune-only)
        echo "[1/1] fine-tune × 60 splits"
        python "${CODE}/run_v3_safe.py" 2>&1 | tee "${LOGS}/finetune.log"
        ;;

    real)
        echo "[1/4] manifest (real)"
        python "${CODE}/build_cases_v3.py" 2>&1 | tee "${LOGS}/build_cases.log"
        echo "[2/4] domain_gap"
        python "${CODE}/domain_gap.py" --device cuda:0 2>&1 | tee "${LOGS}/domain_gap.log"
        echo "[3/4] baselines"
        python "${CODE}/baseline_v3.py" --device cuda:0 2>&1 | tee "${LOGS}/baselines.log"
        echo "[4/4] fine-tune × 60 splits"
        python "${CODE}/run_v3_safe.py" 2>&1 | tee "${LOGS}/finetune.log"
        ;;

    *)
        echo "Usage: $0 [real|dry-run|manifest-only|baselines-only|finetune-only]" >&2
        exit 2
        ;;
esac

echo "[run_v3] done (mode=${MODE})"
