"""V3 cases (100~103) 안전 실행.

- 4 cases × 3 repeats × 5 folds = 60 jobs
- GPU당 동시 1 job, retry 3
- 큰 syn 먼저(V3-A → V3-B → V3-C → V3-Ctrl)
- 이미 완료된 (case, r, f) JSON이 있으면 건너뜀
"""
import os
import sys
import time
import queue
import threading
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path('/home/kyungho2/kh/project_CLIP/2차연구')
MANIFEST = ROOT / 'data/case_manifests/v3_cases.csv'
RESULTS = ROOT / 'results_v3'
LOGS = ROOT / 'logs_v3'
RESULTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


def assert_manifest_real():
    """manifest에 dry-run placeholder가 남아 있으면 실행을 거부."""
    if not MANIFEST.exists():
        raise SystemExit(f'manifest 없음: {MANIFEST}. build_cases_v3.py를 먼저 실행하세요.')
    df = pd.read_csv(MANIFEST)
    bad = df[df['file_path'].str.startswith('[DRY-RUN-PLACEHOLDER]', na=False)]
    if len(bad) > 0:
        raise SystemExit(
            f'manifest에 dry-run placeholder가 {len(bad)}개 있어 실행을 거부합니다.\n'
            f'  → syn_filtered_v3/{{C,CV,others}} 신규 풀 확보 후 build_cases_v3.py를 (--dry-run 없이) 다시 실행하세요.'
        )

CASES = [100, 101, 102, 103]  # V3-Ctrl/A/B/C
REPEATS = [0, 1, 2]
FOLDS = [0, 1, 2, 3, 4]
GPUS = ['cuda:0', 'cuda:1', 'cuda:2']
EPOCHS = 20
BATCH_SIZE = 16  # GPU 공유 환경에서 OOM 방지
MAX_RETRY = 3
# V3-A(101, syn 가장 많음) 먼저, V3-Ctrl(100, syn 없음) 마지막
ORDER = {101: 0, 102: 1, 103: 2, 100: 3}


def jobs_to_run():
    todo = []
    for c in CASES:
        for r in REPEATS:
            for f in FOLDS:
                out = RESULTS / f'case{c}_r{r}_f{f}.json'
                if out.exists():
                    continue
                todo.append((c, r, f, 0))
    todo.sort(key=lambda t: (ORDER[t[0]], t[1], t[2]))
    return todo


def worker(gpu, q, log_lock, stats):
    while True:
        try:
            c, r, f, retry = q.get_nowait()
        except queue.Empty:
            return
        log = LOGS / f'v3_case{c}_r{r}_f{f}_{gpu.replace(":", "")}.log'
        cmd = [
            sys.executable, str(ROOT / 'code' / 'evaluate_v3.py'),
            '--case', str(c), '--repeat', str(r), '--fold', str(f),
            '--device', gpu, '--epochs', str(EPOCHS),
            '--batch_size', str(BATCH_SIZE),
        ]
        t0 = time.time()
        with open(log, 'w') as fh:
            ret = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        dt = time.time() - t0
        out_path = RESULTS / f'case{c}_r{r}_f{f}.json'
        with log_lock:
            if ret.returncode == 0 and out_path.exists():
                stats['ok'] += 1
                print(f'[{gpu}] case={c} r={r} f={f} OK t={dt:.0f}s', flush=True)
            else:
                stats['fail'] += 1
                if retry < MAX_RETRY:
                    print(f'[{gpu}] case={c} r={r} f={f} FAIL (retry {retry+1}) t={dt:.0f}s', flush=True)
                    q.put((c, r, f, retry + 1))
                    time.sleep(15)
                else:
                    print(f'[{gpu}] case={c} r={r} f={f} GIVE UP after {retry+1} tries', flush=True)
        q.task_done()


def main():
    assert_manifest_real()
    todo = jobs_to_run()
    print(f'V3 jobs to run: {len(todo)}', flush=True)
    if not todo:
        print('  (nothing to do — all done)')
        return
    q = queue.Queue()
    for j in todo:
        q.put(j)
    log_lock = threading.Lock()
    stats = {'ok': 0, 'fail': 0}
    threads = [threading.Thread(target=worker, args=(g, q, log_lock, stats), daemon=True)
               for g in GPUS]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f'done. ok={stats["ok"]} fail={stats["fail"]} elapsed={time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
