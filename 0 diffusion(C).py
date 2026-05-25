"""
============================================================
GEN 2 v2: CaCO3 FE-SEM 다형체별 분리 학습 디퓨전 (Path B-light)
============================================================
v1(diffusion_caco3.py)과의 차이점 — 결정 형상 학습 강화:
  - [NEW] Variance threshold crop (std ≥ 35)
          → 부스러기-only 패치 거부, cluster 영역 학습 비중 ↑
          → 5번 재시도 후에도 통과 못 하면 마지막 패치 사용 (학습 멈춤 방지)
  - [NEW] 16×16 attention 추가 (config 기본값)
          → attn_resolutions = [16, 32]
          → 거시 구조(큰 cluster) 학습 능력 강화
  - 출력 폴더: {output_dir}/{label}_model_v2/  (v1 baseline과 분리)

v1과 공유하는 기능:
  - 다형체(A/C/V) 라벨별 분리 학습 (CLI 인자)
  - LR warmup 1,000 step
  - Best moving-average loss 시점 ckpt 자동 저장 (best.pt)
  - PIL 기반 정확한 512x512 grid/individual 저장
  - 3종 CSV 로깅 (train/sample/eval)

실행 예 (Path B-light로 C만 재학습):
  python diffusion_caco3_v2.py --label C --device cuda:1 --total_steps 500000
"""

import os
import glob
import math
import time
import json
import copy
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# Dataset: 정보바 자동 제거 + 512x512 Random Crop + 회전/뒤집기
# ============================================================

def detect_info_bar(img_array, max_scan=200):
    """
    SEM 이미지 하단의 정보 바(스케일·배율 등 메타데이터 영역)를
    행별 평균값 변화로 자동 감지하여 그 높이를 반환.
    이 영역은 학습에 불필요하므로 잘라내야 함.
    """
    h = img_array.shape[0]
    # 이미지 상단 절반은 콘텐츠 영역으로 가정 → 평균 밝기 기준
    content_mean = img_array[: h // 2].mean()
    row_means = img_array.mean(axis=1)

    # 하단부터 위로 스캔하면서 콘텐츠 평균과 충분히 비슷한 첫 행을 탐색
    for i in range(h - 10, h - max_scan, -1):
        block = row_means[i : i + 10]
        if all(abs(m - content_mean) < content_mean * 0.5 for m in block):
            return h - (i + 10) + 5
    return 80  # 감지 실패 시 기본값


class SEMPatchDataset(Dataset):
    """
    1μm 폴더에서 특정 라벨(A/C/V)에 해당하는 파일만 골라 로드한다.
    각 이미지는 정보바 제거 후 메모리에 보관하며,
    __getitem__에서 매번 무작위 512x512 패치를 잘라
    회전·플립·정규화를 적용해 반환한다.
    """

    def __init__(self, image_dir, label, crop_size=512):
        self.crop_size = crop_size
        self.images = []

        # 파일명 prefix가 "{label}-"로 시작하는 파일만 선택
        # 예: label="A" → A-25-3.tif, A-26-3.tif, ... 만 매칭
        # C+V, others는 자연스럽게 제외됨
        prefix = f"{label}-"
        all_paths = []
        for ext in ["*.tif", "*.TIF", "*.png", "*.PNG", "*.jpg", "*.jpeg"]:
            all_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        paths = sorted(
            p for p in set(all_paths) if os.path.basename(p).startswith(prefix)
        )

        print(f"[Filter] '{prefix}*' 패턴으로 {len(paths)}개 파일 매칭")

        # 각 파일을 그레이스케일로 읽고 정보바 자르고 보관
        for path in paths:
            img = np.array(Image.open(path).convert("L"), dtype=np.float32)
            h, w = img.shape
            bar_h = detect_info_bar(img.astype(np.uint8))
            img = img[: h - bar_h, :]

            # crop_size보다 작으면 학습 불가하므로 제외
            if img.shape[0] >= crop_size and img.shape[1] >= crop_size:
                self.images.append(img)
                name = os.path.basename(path)
                print(f"  {name:20s} → 정보바 {bar_h}px 제거 → {img.shape[1]}x{img.shape[0]}")
            else:
                print(f"  {os.path.basename(path):20s} → 크기 부족, 제외")

        print(f"[Dataset] '{label}' 라벨 {len(self.images)}장 로드 완료")

        if len(self.images) == 0:
            raise RuntimeError(f"라벨 '{label}'에 해당하는 이미지가 없습니다.")

    def __len__(self):
        # 한 epoch에 충분한 step이 돌도록 가상 길이를 200배로 부풀림
        # (실제로는 매번 무작위 패치를 추출하므로 무한정 사용 가능)
        return len(self.images) * 200

    def __getitem__(self, idx):
        # 이미지 한 장 선택
        img = self.images[idx % len(self.images)]
        h, w = img.shape

        # ───── [v2 NEW] Variance threshold crop ─────
        # 부스러기-only 패치 거부, cluster 영역 학습 비중 ↑
        # std ≥ 35: C 이미지에서 하위 ~30% 부스러기 영역만 거름 (실측 분포 기반)
        # 5번 재시도 후에도 통과 못 하면 마지막 패치 사용 (학습 멈춤 방지)
        for _ in range(5):
            top = np.random.randint(0, h - self.crop_size + 1)
            left = np.random.randint(0, w - self.crop_size + 1)
            patch = img[top : top + self.crop_size, left : left + self.crop_size]
            if patch.std() >= 35:  # 결정 풍부 패치만 통과
                break
        # 5회 재시도 실패 시: patch에 마지막 시도값이 들어있으므로 그대로 사용

        # 90도 회전 4가지 중 하나 무작위
        k = np.random.randint(0, 4)
        patch = np.rot90(patch, k).copy()

        # 좌우/상하 50% 확률로 뒤집기
        if np.random.random() > 0.5:
            patch = np.fliplr(patch).copy()
        if np.random.random() > 0.5:
            patch = np.flipud(patch).copy()

        # 픽셀값 [0, 255] → [-1, 1] 정규화 (디퓨전 표준)
        patch = patch / 127.5 - 1.0
        return torch.FloatTensor(patch).unsqueeze(0)


# ============================================================
# Sinusoidal Time Embedding
# 디퓨전 모델은 매 timestep마다 동작이 달라야 하므로
# step 번호를 sin/cos 기반 벡터로 변환해 모델에 입력
# ============================================================

def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    emb = t[:, None].float() * freqs[None]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ============================================================
# UNet 빌딩 블록
# ============================================================

class ResBlock(nn.Module):
    """
    Residual Block: GroupNorm → SiLU → Conv → 시간 임베딩 더하기
    → GroupNorm → SiLU → Dropout → Conv → skip 연결
    디퓨전 UNet의 가장 기본 단위.
    """

    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """
    저해상도 feature map에서 전역 관계 학습용 멀티헤드 어텐션.
    SEM 결정의 공간 배치(전역) 정보를 모델이 인식하게 함.
    32x32 해상도에만 적용 (계산량 O(N^2) 때문).
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = self.norm(x).view(b, c, h * w)
        qkv = self.qkv(x_flat).view(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv.unbind(1)
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v).reshape(b, c, h * w)
        return self.proj(out).view(b, c, h, w) + x


class Downsample(nn.Module):
    """stride=2 컨볼루션으로 해상도 절반으로 축소."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """nearest 보간으로 해상도 2배 확대 후 컨볼루션."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ============================================================
# UNet 본체 (5단계, 32x32 해상도에 어텐션)
# ============================================================

class UNet(nn.Module):
    """
    512x512 → 256 → 128 → 64 → 32 (다운샘플 4회) 구조.
    채널 수는 base_ch * ch_mults로 단계별 증가.
    Encoder–Middle–Decoder + Skip connection.
    """

    def __init__(self, in_ch=1, base_ch=64, ch_mults=(1, 2, 2, 4, 4),
                 attn_resolutions=(32,), num_res_blocks=2, dropout=0.1):
        super().__init__()
        self.base_ch = base_ch
        self.num_levels = len(ch_mults)
        self.num_res_blocks = num_res_blocks
        emb_dim = base_ch * 4

        # 시간 임베딩 MLP (sinusoidal → MLP)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        # 입력 채널을 base_ch로 변환
        self.input_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ───── Encoder (해상도 점진적 축소) ─────
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        chs = [base_ch]            # skip 연결 보관용 채널 기록
        now_ch = base_ch
        now_res = 512

        for level, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(now_ch, out_ch, emb_dim, dropout)])
                if now_res in attn_resolutions:
                    block.append(SelfAttention(out_ch))
                self.enc_blocks.append(block)
                now_ch = out_ch
                chs.append(now_ch)
            if level < len(ch_mults) - 1:
                self.downsamples.append(Downsample(now_ch))
                chs.append(now_ch)
                now_res //= 2

        # ───── Middle (가장 저해상도, attention 포함) ─────
        self.mid1 = ResBlock(now_ch, now_ch, emb_dim, dropout)
        self.mid_attn = SelfAttention(now_ch)
        self.mid2 = ResBlock(now_ch, now_ch, emb_dim, dropout)

        # ───── Decoder (해상도 점진적 복원, skip 연결 활용) ─────
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                block = nn.ModuleList([ResBlock(now_ch + skip_ch, out_ch, emb_dim, dropout)])
                if now_res in attn_resolutions:
                    block.append(SelfAttention(out_ch))
                self.dec_blocks.append(block)
                now_ch = out_ch
            if level > 0:
                self.upsamples.append(Upsample(now_ch))
                now_res *= 2

        # ───── 출력 (zero-init: 학습 초기 residual=0) ─────
        self.out_norm = nn.GroupNorm(32, now_ch)
        self.out_conv = nn.Conv2d(now_ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t):
        emb = sinusoidal_embedding(t, self.base_ch)
        emb = self.time_mlp(emb)

        h = self.input_conv(x)
        skips = [h]

        # Encoder pass
        enc_idx = 0
        for level in range(self.num_levels):
            for _ in range(self.num_res_blocks):
                layers = self.enc_blocks[enc_idx]
                h = layers[0](h, emb)
                for layer in layers[1:]:
                    h = layer(h)
                skips.append(h)
                enc_idx += 1
            if level < self.num_levels - 1:
                h = self.downsamples[level](h)
                skips.append(h)

        # Middle
        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)

        # Decoder pass (skip concat)
        dec_idx = 0
        us_idx = 0
        for level in range(self.num_levels):
            for _ in range(self.num_res_blocks + 1):
                h = torch.cat([h, skips.pop()], dim=1)
                layers = self.dec_blocks[dec_idx]
                h = layers[0](h, emb)
                for layer in layers[1:]:
                    h = layer(h)
                dec_idx += 1
            if level < self.num_levels - 1:
                h = self.upsamples[us_idx](h)
                us_idx += 1

        return self.out_conv(F.silu(self.out_norm(h)))


# ============================================================
# Cosine Noise Schedule
# 미세 텍스처(SEM 결정 표면)를 보존하기에 유리한 스케줄
# ============================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999)


# ============================================================
# 이미지 저장 유틸 (PIL 기반, 정확한 512x512 해상도 보존)
# ============================================================

def save_grid_pil(imgs_uint8, save_path, rows, cols, cell_size=512):
    """
    [0,255] uint8 numpy 이미지 리스트를 받아
    정확히 (cols*cell_size) x (rows*cell_size) 픽셀의
    grid PNG 파일로 저장. matplotlib 여백 없음.
    """
    grid = np.zeros((rows * cell_size, cols * cell_size), dtype=np.uint8)
    for i, img in enumerate(imgs_uint8):
        if i >= rows * cols:
            break
        r, c = i // cols, i % cols
        grid[r * cell_size:(r + 1) * cell_size,
             c * cell_size:(c + 1) * cell_size] = img
    Image.fromarray(grid, mode="L").save(save_path)


def save_individual_pil(img_uint8, save_path):
    """단일 grayscale 이미지를 정확한 해상도로 PNG 저장."""
    Image.fromarray(img_uint8, mode="L").save(save_path)


# ============================================================
# Trainer (DDPM + DDIM + AMP + LR warmup + Best ckpt + CSV log)
# ============================================================

class DiffusionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.num_timesteps = config["num_timesteps"]
        self.label = config["label"]

        # ───── Noise schedule (forward process 계수) 미리 계산 ─────
        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)
        self.sqrt_ac = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_ac = torch.sqrt(1.0 - self.alphas_cumprod)

        # ───── 모델 + EMA 모델 ─────
        self.model = UNet(
            in_ch=1,
            base_ch=config["base_ch"],
            ch_mults=tuple(config["ch_mults"]),
            attn_resolutions=tuple(config["attn_resolutions"]),
            num_res_blocks=config["num_res_blocks"],
            dropout=config["dropout"],
        ).to(self.device)

        # EMA(지수이동평균) 모델: 학습 가중치의 부드러운 평균
        # 디퓨전에서 샘플 품질이 EMA 가중치에서 크게 향상됨
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()

        # ───── 최적화기 + AMP ─────
        # AdamW: 표준 weight_decay 적용 가능한 옵티마이저
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config["lr"], weight_decay=1e-4
        )
        # GradScaler: fp16 학습 시 gradient underflow 방지
        self.scaler = torch.amp.GradScaler("cuda")

        # ───── 출력 디렉토리 (라벨별 분리, v2 suffix) ─────
        # 예: /home/namhu/철강부산물/GEN 2/diffusion_results/C_model_v2/
        self.output_dir = Path(config["output_dir"]) / f"{self.label}_model_v2"
        (self.output_dir / "samples").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "individual").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        # ───── CSV 로깅 파일 초기화 ─────
        # train/sample CSV는 학습 중 채워지고, eval CSV는 학습 후 evaluate.py가 채움.
        # 셋 모두 헤더는 학습 시작 시점에 미리 생성하여 디렉토리 구조 일관성 유지.
        self.train_csv = self.output_dir / "train_metrics.csv"
        self.sample_csv = self.output_dir / "sample_metrics.csv"
        self.eval_csv = self.output_dir / "eval_metrics.csv"

        # 주의: moving_avg_loss는 최근 1000 step의 단순 이동평균이며,
        #       EMA "모델"의 loss와는 무관함 (혼동 방지를 위해 ema_loss → moving_avg_loss)
        self._init_csv(self.train_csv,
                       ["step", "loss", "moving_avg_loss", "lr", "grad_norm", "elapsed_sec"])
        self._init_csv(self.sample_csv,
                       ["step", "sample_time_sec", "mean_pixel", "std_pixel", "sample_dir"])
        # eval CSV는 evaluate.py에서 ckpt별로 한 줄씩 append됨.
        # 만약 evaluate.py 작성 시 컬럼 셋이 바뀌면 그쪽에서 헤더를 덮어쓸 것.
        self._init_csv(self.eval_csv,
                       ["ckpt", "fid", "dinov2_fid", "kid", "density", "coverage",
                        "glcm_jsd", "fft_l2", "psd_wasserstein", "aug_boost"])

        # ───── Best moving-avg loss 추적용 변수 ─────
        # 최근 1000 step의 이동평균 loss가 최저인 시점의 ckpt를 best.pt로 저장
        # (EMA 모델의 loss가 아니라 학습 손실의 이동평균임)
        self.best_moving_avg_loss = float("inf")
        self.best_step = 0

        # ───── 모델 파라미터 수 출력 ─────
        params = sum(p.numel() for p in self.model.parameters())
        print(f"[Model] UNet: {params:,} params")
        print(f"[Output] {self.output_dir}")

    def _init_csv(self, path, header):
        """CSV 파일을 만들고 헤더 한 줄을 쓴다 (이미 있으면 보존)."""
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def _append_csv(self, path, row):
        """CSV 파일에 한 줄 추가."""
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def get_warmup_lr(self, step, base_lr, warmup_steps):
        """
        학습 초기 안정성을 위한 LR warmup.
        step 0~warmup_steps 동안 lr을 0 → base_lr로 선형 증가.
        그 이후엔 base_lr 유지.
        """
        if step < warmup_steps:
            return base_lr * step / warmup_steps
        return base_lr

    def q_sample(self, x0, t, noise):
        """
        Forward diffusion: x_t = sqrt(α_t)·x_0 + sqrt(1-α_t)·ε
        깨끗한 이미지 x0에 noise를 timestep t만큼 섞는다.
        """
        return (self.sqrt_ac[t].view(-1, 1, 1, 1) * x0
                + self.sqrt_one_minus_ac[t].view(-1, 1, 1, 1) * noise)

    @torch.no_grad()
    def ema_update(self, decay=0.9999):
        """매 step 후 EMA 모델 가중치를 업데이트 (lerp = 선형 보간)."""
        for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.lerp_(p, 1 - decay)

    @torch.no_grad()
    def ddim_sample(self, num_samples=16, ddim_steps=50):
        """
        DDIM 샘플링: T=1000을 ddim_steps(=50)으로 압축해
        이미지를 빠르게 생성 (DDPM 1000-step 대비 20배 빠름).
        반환: [-1, 1] 범위의 텐서 (batch, 1, 512, 512)
        """
        self.ema_model.eval()
        shape = (num_samples, 1, 512, 512)
        c = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, c))

        # 가우시안 노이즈에서 시작
        x = torch.randn(shape, device=self.device)

        # 역방향(t=999 → 0)으로 노이즈를 점진적으로 제거
        for i in reversed(range(len(timesteps))):
            t = timesteps[i]
            t_prev = timesteps[i - 1] if i > 0 else -1
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

            with torch.amp.autocast("cuda"):
                noise_pred = self.ema_model(x, t_batch)

            at = self.alphas_cumprod[t]
            at_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

            # 예측된 x0 추정 후 다음 timestep으로 이동 (DDIM deterministic)
            x0_pred = (x - torch.sqrt(1 - at) * noise_pred) / torch.sqrt(at)
            x0_pred = x0_pred.clamp(-1, 1)
            x = torch.sqrt(at_prev) * x0_pred + torch.sqrt(1 - at_prev) * noise_pred

        return x.clamp(-1, 1)

    def save_samples(self, step, num=16):
        """
        DDIM으로 num개 이미지를 생성한 뒤
        - 4x4 grid 1장 (PIL, 정확한 512x512×4×4)
        - 개별 이미지 num장 (PIL, 정확한 512x512)
        를 저장하고 sample_metrics.csv에 한 줄을 기록한다.
        """
        sample_t0 = time.time()
        images = self.ddim_sample(num)
        sample_time = time.time() - sample_t0

        # [-1, 1] → [0, 255] uint8로 변환
        imgs = (images.cpu().squeeze(1).numpy() + 1) / 2  # → [0,1]
        imgs = np.clip(imgs, 0, 1)
        imgs_u8 = (imgs * 255).astype(np.uint8)

        # ───── grid 저장 (PIL, 매트플롯 여백 없음) ─────
        rows = int(math.ceil(math.sqrt(num)))
        cols = int(math.ceil(num / rows))
        grid_path = self.output_dir / "samples" / f"step_{step:06d}.png"
        save_grid_pil(imgs_u8, grid_path, rows, cols, cell_size=512)

        # ───── 개별 이미지 저장 (PIL) ─────
        ind_dir = self.output_dir / "individual"
        for i in range(num):
            save_individual_pil(
                imgs_u8[i],
                ind_dir / f"step_{step:06d}_{i + 1:02d}.png",
            )

        # ───── sample_metrics.csv 기록 ─────
        # 평균/표준편차로 분포가 시간에 따라 안정화되는지 모니터
        self._append_csv(self.sample_csv, [
            step,
            f"{sample_time:.2f}",
            f"{imgs.mean():.4f}",
            f"{imgs.std():.4f}",
            str(ind_dir),
        ])

    def save_checkpoint(self, step, name=None):
        """
        체크포인트 저장 (모델/EMA/옵티마이저/스케일러/현재 config 모두 포함).
        name이 지정되면 그 이름으로(예: "best.pt"), 아니면 step 번호로 저장.
        """
        ckpt_name = name if name else f"ckpt_{step:06d}.pt"
        torch.save(
            {
                "step": step,
                "model": self.model.state_dict(),
                "ema_model": self.ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config": self.config,
                "best_moving_avg_loss": self.best_moving_avg_loss,
                "best_step": self.best_step,
            },
            self.output_dir / "checkpoints" / ckpt_name,
        )

    def train(self, dataloader, total_steps):
        """학습 메인 루프."""
        max_seconds = self.config.get("max_hours", 62) * 3600
        warmup_steps = self.config.get("warmup_steps", 1000)
        base_lr = self.config["lr"]

        print(f"[Train] {total_steps:,} steps (max), batch_size={self.config['batch_size']}")
        print(f"[Time]  최대 {self.config.get('max_hours', 62)}시간 후 자동 종료")
        print(f"[Device] {self.device}")
        print(f"[AMP] Mixed precision 활성화")
        print(f"[Warmup] {warmup_steps} step 동안 0 → {base_lr:.1e}")

        data_iter = iter(dataloader)
        losses = []
        start_time = time.time()

        self.model.train()
        pbar = tqdm(range(1, total_steps + 1), desc=f"Training[{self.label}]")

        for step in pbar:
            # ───── 시간 한도 체크 ─────
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                print(f"\n[Time Limit] {elapsed / 3600:.1f}시간 경과. Step {step}에서 종료.")
                break

            # ───── LR warmup 적용 ─────
            current_lr = self.get_warmup_lr(step, base_lr, warmup_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = current_lr

            # ───── 미니배치 가져오기 ─────
            try:
                x0 = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x0 = next(data_iter)
            x0 = x0.to(self.device, non_blocking=True)

            # ───── Forward diffusion: 무작위 t에서 노이즈 추가 ─────
            t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=self.device)
            noise = torch.randn_like(x0)
            xt = self.q_sample(x0, t, noise)

            # ───── 노이즈 예측 + MSE 손실 (AMP) ─────
            with torch.amp.autocast("cuda"):
                noise_pred = self.model(xt, t)
                loss = F.mse_loss(noise_pred, noise)

            # ───── Backward + gradient clipping + step ─────
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ───── EMA 가중치 업데이트 ─────
            self.ema_update()
            losses.append(loss.item())

            # ───── 첫 100 step 후 학습 속도 추정 출력 ─────
            if step == 100:
                speed = 100 / (time.time() - start_time)
                est_hours = total_steps / speed / 3600
                print(f"\n[Speed] {speed:.1f} steps/sec → 예상: {est_hours:.1f}시간")

            # ───── 진행바 갱신 + train_metrics.csv 기록 (매 100 step) ─────
            if step % 100 == 0:
                # avg_loss        : 최근 100 step 학습 손실의 단순 평균
                # moving_avg_loss : 최근 1000 step 학습 손실의 단순 이동평균
                #                   (EMA "모델"의 loss가 아니라, 학습 noise를 평탄화한 추세)
                avg_loss = float(np.mean(losses[-100:]))
                moving_avg_loss = float(np.mean(losses[-1000:])) if len(losses) >= 100 else avg_loss
                pbar.set_postfix(
                    loss=f"{avg_loss:.5f}", lr=f"{current_lr:.1e}", t=f"{elapsed/60:.0f}m"
                )
                self._append_csv(self.train_csv, [
                    step,
                    f"{avg_loss:.6f}",
                    f"{moving_avg_loss:.6f}",
                    f"{current_lr:.6e}",
                    f"{float(grad_norm):.4f}",
                    f"{elapsed:.1f}",
                ])

                # ───── Best moving-avg loss 갱신 시 best.pt 저장 ─────
                # warmup 종료 + 충분히 학습된 후(step 5000+)부터 추적
                if step >= 5000 and moving_avg_loss < self.best_moving_avg_loss:
                    self.best_moving_avg_loss = moving_avg_loss
                    self.best_step = step
                    self.save_checkpoint(step, name="best.pt")

            # ───── 주기적 샘플링 (모니터링용 16장) ─────
            if step % self.config["sample_interval"] == 0:
                self.save_samples(step)
                self.model.train()  # ddim_sample이 model을 eval로 두지 않지만 안전 차원

            # ───── 주기적 체크포인트 저장 ─────
            if step % self.config["ckpt_interval"] == 0:
                self.save_checkpoint(step)
                torch.cuda.empty_cache()

        # ───── 최종 저장 ─────
        self.save_checkpoint(step)
        self.save_samples(step, num=16)

        # ───── 손실 곡선 (간단 시각화) ─────
        # 자세한 곡선은 visualize.py에서 train_metrics.csv 기반으로 별도 생성
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses, alpha=0.2, color="blue", label="raw loss")
        window = min(500, len(losses) // 5 + 1)
        if window > 1:
            kernel = np.ones(window) / window
            ax.plot(np.convolve(losses, kernel, mode="valid"), color="blue",
                    linewidth=2, label="smoothed")
        ax.set_title(f"Training Loss (MSE) - {self.label} model")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "loss_curves.png", dpi=150)
        plt.close()

        total_time = time.time() - start_time
        print(f"\n[Done] {total_time / 60:.1f}분 소요, {step:,} steps 완료")
        print(f"[Best] moving-avg loss={self.best_moving_avg_loss:.5f} @ step {self.best_step}")

        # ───── config 백업 ─────
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False, default=str)

        return losses


# ============================================================
# Main 실행부
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GEN 2 CaCO3 다형체별 분리 학습 디퓨전"
    )
    parser.add_argument("--label", type=str, required=True, choices=["A", "C", "V"],
                        help="학습할 다형체 라벨 (A=Aragonite, C=Calcite, V=Vaterite)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="GPU 디바이스 (예: cuda:0, cuda:1, cuda:2)")
    parser.add_argument("--image_dir", type=str,
                        default="/home/namhu/철강부산물/GEN 2/CaCO3_SEM/1μm",
                        help="원본 이미지 폴더 경로")
    parser.add_argument("--output_dir", type=str,
                        default="/home/namhu/철강부산물/GEN 2/diffusion_results",
                        help="결과 저장 상위 디렉토리 (라벨별로 자동 분리됨)")
    parser.add_argument("--total_steps", type=int, default=500000)  # v2: 500k 기본
    parser.add_argument("--batch_size", type=int, default=4)
    # dry-run을 위해 sample/ckpt 주기도 인자로 받음 (기본값: 본격 학습용)
    parser.add_argument("--sample_interval", type=int, default=2000)
    parser.add_argument("--ckpt_interval", type=int, default=10000)
    parser.add_argument("--max_hours", type=float, default=80.0)  # v2: 500k step 대비 여유
    args = parser.parse_args()

    # cudnn 벤치마크 모드: 입력 크기 고정 시 컨볼루션 최적 알고리즘 자동 선택
    torch.backends.cudnn.benchmark = True

    # 학습 설정 (Path B-light: attn_resolutions에 16 추가)
    config = {
        "label": args.label,
        "base_ch": 64,
        "ch_mults": [1, 2, 2, 4, 4],
        "attn_resolutions": [16, 32],     # [v2 NEW] 16x16 attention 추가 → 거시 구조 학습
        "num_res_blocks": 2,
        "dropout": 0.1,
        "num_timesteps": 1000,
        "lr": 1e-4,
        "warmup_steps": 1000,
        "batch_size": args.batch_size,
        "total_steps": args.total_steps,  # v2 기본 500k
        "max_hours": args.max_hours,      # v2 기본 80h
        "device": args.device,
        "sample_interval": args.sample_interval,
        "ckpt_interval": args.ckpt_interval,
        "image_dir": args.image_dir,
        "output_dir": args.output_dir,
        "variance_threshold": 35,         # [v2 NEW] cluster 영역 우선 크롭 임계값
    }

    print("=" * 60)
    print(f"GEN 2 v2 (Path B-light) | DDPM + DDIM")
    print(f"  Label='{args.label}' | Device={args.device}")
    print(f"  Variance threshold: std ≥ {config['variance_threshold']}")
    print(f"  Attention: {config['attn_resolutions']}")
    print(f"  Total steps: {config['total_steps']:,}, Max hours: {config['max_hours']}")
    print("=" * 60)

    # ───── 데이터셋 로드 (라벨 필터링 적용) ─────
    dataset = SEMPatchDataset(args.image_dir, label=args.label, crop_size=512)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    # ───── 학습 시작 ─────
    trainer = DiffusionTrainer(config)
    losses = trainer.train(dataloader, config["total_steps"])
