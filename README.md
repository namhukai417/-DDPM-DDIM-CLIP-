# -DDPM-DDIM-CLIP-
순서 / 파일 / 역할 분류
0 diffusion(A, V, C+V, others).py  이미지 합성 \n
0 diffusion(C).py  이미지 합성\n
1  filter_syn_quality.py   데이터 준비 (합성풀 C·C+V·others)  
1  filter_syn_quality.py   데이터 준비 (합성풀 A·V)        
2  build_cases_v3.py     핵심 — K-fold split + manifest  
3  domain_gap.py        보조 진단 (OG↔Syn 격차, 분류 본실험 아님)    
4  baseline_v3.py         보조 비교 (majority·zero-shot·linear-probe 기준선)
5  evaluate_v3.py         ★ 본 실험 — fine-tune + 5-class 평가 엔진
5  run_v3_safe.py         ★ 본 실험 — 60-job 오케스트레이터
6  aggregate_v3.py      집계 — 60개 결과 요약·통계검정
