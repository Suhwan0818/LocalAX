"""ex32_mini_gpt_sol.py — 미니 GPT 조립·학습 (정답)

- 대응 슬라이드: M2 '미니 GPT 조립 ① 부품 선언 · ② forward' / M2 실습 2 '부품 조립 후 1500스텝 학습' · 'TODO 5개'
- 내용
  TODO(1) 학습 루프: 100스텝마다 val loss
  TODO(2) get_batch: 한 칸 밀린 y
  TODO(3) GPT.forward: 임베딩 → 블록 → ln_f → lm_head → CE loss
  TODO(4) Block.forward: 잔차 두 번 (x + attn(norm(x)), x + ffn(norm(x)))
  TODO(5) generate: model.eval · idx[:, -T:]
  선택 과제: wpe → RoPE 교체 · RMSNorm ↔ LayerNorm
- 기호: B: 배치 크기(문장 수) · T: 토큰 수(시퀀스 길이 · 최대 --T) · C: 임베딩 차원 · H: 헤드 수 · hs: 헤드 하나의 차원(C/H)
        V: 어휘 크기(글자 종류 수) · L: 블록 수(층 수) · d_ff: FFN 중간 차원 · (..., T, hs): 앞쪽 축은 무엇이든(B·H 등) 마지막 두 축만
        A: 어텐션 가중치(softmax 결과 · 행 합 1) · CE: 교차 엔트로피 · ln V: 균등 추측 손실(V개 중 찍기)
- 설정: T=64 · C=128 · H=4 · L=4 · d_ff=512(SwiGLU) — V=65면 약 1.07M, 한국어 문자 어휘면 V가 커서 재계산됨
- 데이터: data/tinystories_ko_sample.txt (문자 단위)
- 실행: pyc ex32_mini_gpt_sol.py --steps 1500 --out results/ex32.json
- 산출물: results/ex32.json · results/ex32_loss.png
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent


# ---- 실습 1(ex31)의 완성본 복제 ----
def attn(q, k, v):                                       # (..., T, hs)
    T, hs = q.shape[-2:]
    s = q @ k.transpose(-2, -1) / math.sqrt(hs)          # ① 점수 · ② 스케일
    m = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    s = s.masked_fill(~m, float("-inf"))                   # 미래 칸 -inf (softmax 전)
    # ③ softmax(행마다): 행 최댓값 빼기 → exp → 행 합으로 나누기
    e = torch.exp(s - s.amax(dim=-1, keepdim=True))       # 최댓값 빼도 결과 같음 · exp 넘침 방지 · exp(-inf) = 0
    a = e / e.sum(dim=-1, keepdim=True)                   # 가중치 A · 행 합 1
    return a @ v                                          # ④ 가중합


class CausalMHA(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        assert C % H == 0
        self.H = H
        self.qkv, self.proj = nn.Linear(C, 3 * C), nn.Linear(C, C)

    def forward(self, x):                                # (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.H, -1).transpose(1, 2)    # (B, T, C) → (B, T, H, hs) → (B, H, T, hs)
        k = k.view(B, T, self.H, -1).transpose(1, 2)
        v = v.view(B, T, self.H, -1).transpose(1, 2)
        y = attn(q, k, v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, C, d_ff):
        super().__init__()
        self.w1, self.w3, self.w2 = nn.Linear(C, d_ff, bias=False), nn.Linear(C, d_ff, bias=False), nn.Linear(d_ff, C, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, C, H, d_ff):
        super().__init__()
        self.ln1, self.attn, self.ln2, self.ffn = nn.RMSNorm(C), CausalMHA(C, H), nn.RMSNorm(C), SwiGLU(C, d_ff)

    def forward(self, x):
        # TODO-BEGIN(4): Pre-Norm 잔차 두 번
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))
        # TODO-END


class GPT(nn.Module):
    def __init__(self, V, T, C, H, L, d_ff):
        super().__init__()
        self.T = T
        self.wte, self.wpe = nn.Embedding(V, C), nn.Embedding(T, C)
        blocks = []
        for _ in range(L):                                           # Block L개 차례로 쌓기
            blocks.append(Block(C, H, d_ff))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f, self.lm_head = nn.RMSNorm(C), nn.Linear(C, V, bias=False)
        self.wte.weight = self.lm_head.weight                        # weight tying

    def forward(self, idx, targets=None):                            # (B, T)
        # TODO-BEGIN(3): 임베딩 → 블록 → ln_f → lm_head → (targets 있으면) CE loss
        pos = torch.arange(idx.size(1), device=idx.device)          # 위치 0, 1, …, T-1
        x = self.wte(idx) + self.wpe(pos)                            # (B, T, C)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))                          # (B, T, V)
        loss = None                                                  # 생성 때는 정답 없음 → loss 없음
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())   # (B·T, V) · (B·T,)
        return logits, loss
        # TODO-END

    @torch.no_grad()
    def generate(self, idx, n: int, temperature: float = 1.0):
        # TODO-BEGIN(5): eval 모드 · 최근 T개만 잘라 넣기 · 마지막 위치 분포에서 샘플링
        self.eval()
        for _ in range(n):
            logits, _ = self(idx[:, -self.T:])                       # 최근 T개만 (wpe 범위 안)
            probs = (logits[:, -1] / temperature).softmax(-1)        # (B, T, V) → 마지막 위치 (B, V)
            nxt = torch.multinomial(probs, 1)                        # 토큰 1개 샘플링 (B, 1)
            idx = torch.cat([idx, nxt], dim=1)                       # idx 뒤에 붙이기
        self.train()
        return idx
        # TODO-END


def get_batch(data: torch.Tensor, B: int, T: int, dev: str):
    # TODO-BEGIN(2): 임의 시작점 B개 → x = data[i:i+T], y = data[i+1:i+T+1] (한 칸 밀림)
    ix = torch.randint(len(data) - T - 1, (B,))                     # 시작점 i를 B개 무작위로
    xs, ys = [], []
    for i in ix:
        xs.append(data[i:i + T])                                    # 입력 (T,)
        ys.append(data[i + 1:i + T + 1])                            # 한 칸 밀린 정답 (T,)
    x = torch.stack(xs)                                             # (B, T)
    y = torch.stack(ys)
    return x.to(dev), y.to(dev)
    # TODO-END


@torch.no_grad()
def estimate(model, data, B, T, dev, iters=20) -> float:
    model.eval()
    total = 0.0
    for _ in range(iters):                                          # 배치 iters개 손실 평균
        _, loss = model(*get_batch(data, B, T, dev))
        total += loss.item()
    model.train()
    return total / iters


# ---------- wrapper 이름표 ----------
# 실제 동작: setup_ 준비 · run_ 실행 · save_ 저장
# 디버깅·로그·시각화: debug_ 검증 · log_ 화면 출력 · viz_ 그래프
def main() -> None:
    a = setup_args()
    setup_seed(a)
    dev = setup_device()
    V, stoi, chars, train_d, val_d, text = setup_data(a)
    model, n_params = setup_model(a, dev, V)
    log_model_info(dev, V, n_params)
    hist, train_sec = run_train(a, dev, train_d, val_d, model) #TODO(1,2,3,4)
    sample, gen_sec = run_generate(dev, stoi, chars, text, model) #TODO(3,4,5)
    log_sample(sample)
    best = run_find_best_val(hist)
    log_read_guide(V, best)
    out = save_results(a, dev, V, n_params, hist, train_sec, gen_sec, sample, best)
    png = viz_loss_curve(V, hist, out)
    log_saved(out, png)


def setup_args() -> argparse.Namespace:
    """실행 인자 읽기"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-chars", type=int, default=1_000_000, help="학습에 쓸 앞부분 글자 수")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="results/ex32.json")
    a = ap.parse_args()
    return a


def setup_seed(a) -> None:
    """난수 seed 고정"""
    torch.manual_seed(a.seed)


def setup_device() -> str:
    """GPU가 있으면 GPU 사용"""
    dev = "cpu"
    if torch.cuda.is_available():
        dev = "cuda"
    return dev


def setup_data(a) -> tuple:
    """글자 어휘 · 토큰 ID 열 · 학습/검증 분할"""
    text = (HERE / "data" / "tinystories_ko_sample.txt").read_text(encoding="utf-8")[: a.n_chars]
    chars = sorted(set(text))                                       # 어휘 = 글자 종류 (정렬 → id 고정)
    V = len(chars)
    stoi = {}
    for i, c in enumerate(chars):                                   # 글자 → id
        stoi[c] = i
    ids = []
    for c in text:
        ids.append(stoi[c])
    data = torch.tensor(ids, dtype=torch.long)                      # 토큰 ID 1차원 열
    n = int(0.9 * len(data))                                        # 앞 90% 학습 · 뒤 10% 검증
    train_d, val_d = data[:n], data[n:]
    return V, stoi, chars, train_d, val_d, text


def setup_model(a, dev, V) -> tuple:
    """GPT 모델 생성 · 파라미터 수 세기"""
    model = GPT(V, a.T, a.C, a.H, a.L, a.d_ff).to(dev)
    n_params = 0
    for p in model.parameters():                                    # 파라미터 텐서마다 원소 수(numel) 더하기
        n_params += p.numel()
    return model, n_params


def log_model_info(dev, V, n_params) -> None:
    """장치 · 어휘 크기 · 파라미터 수 · 균등 추측 손실 출력"""
    print(f"device={dev} · V={V} · 파라미터 {n_params / 1e6:.3f}M (V=65 기준이면 약 1.07M)")
    print(f"균등 추측 손실 ln V = {math.log(V):.2f}")


def run_train(a, dev, train_d, val_d, model) -> tuple:
    """steps회 학습 → 손실 기록 · 학습 시간"""
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    hist = []
    t0 = time.time()
    # TODO-BEGIN(1): steps회 학습 · 100스텝(및 0·마지막)마다 train/val 손실 기록
    for step in range(a.steps + 1):
        if step % 100 == 0 or step == a.steps:
            tr = estimate(model, train_d, a.batch, a.T, dev)        # train 먼저, val 다음
            va = estimate(model, val_d, a.batch, a.T, dev)
            hist.append({"step": step, "train": round(tr, 4), "val": round(va, 4)})
            print(f"step {step:>5}  train {tr:.3f}  val {va:.3f}  ({time.time() - t0:.0f}s)")
        if step == a.steps:
            break
        _, loss = model(*get_batch(train_d, a.batch, a.T, dev))
        opt.zero_grad(set_to_none=True)                             # 기울기 초기화
        loss.backward()                                             # 역전파
        opt.step()                                                  # 가중치 갱신
    # TODO-END
    train_sec = round(time.time() - t0, 1)                          # 학습 구간만 · 생성 시간은 gen_sec로 따로
    return hist, train_sec


def run_generate(dev, stoi, chars, text, model) -> tuple:
    """본문 앞 10자를 접두로 300자 생성 → 샘플 · 생성 시간"""
    t1 = time.time()
    prefix = text[:10]
    prefix_ids = []
    for c in prefix:
        prefix_ids.append(stoi[c])
    idx = torch.tensor([prefix_ids], device=dev)                    # (1, 접두 글자 수)
    gen_ids = model.generate(idx, 300)[0].tolist()                  # 접두 + 생성 300개 id #TODO(3,4,5)
    parts = []
    for i in gen_ids:                                               # id → 글자
        parts.append(chars[i])
    sample = "".join(parts)
    gen_sec = round(time.time() - t1, 1)
    return sample, gen_sec


def log_sample(sample) -> None:
    """생성 샘플 출력"""
    print("\n생성 샘플 300자:\n" + sample)


def run_find_best_val(hist) -> dict:
    """손실 기록에서 val 최저 기록 찾기"""
    best = hist[0]
    for h in hist:                                                  # val 최저 기록 ('<'라서 동률이면 앞 step 유지 · min과 같음)
        if h["val"] < best["val"]:
            best = h
    return best


def log_read_guide(V, best) -> None:
    """ln V · val 최저 step · 읽는 법 출력"""
    print(f"\n읽는 법: step 0 손실 ≈ ln V({math.log(V):.2f})에서 출발하는지 · val 최저 step {best['step']} · val이 최저점을 지나 다시 오르기 시작하면 과적합")


def save_results(a, dev, V, n_params, hist, train_sec, gen_sec, sample, best) -> Path:
    """실행 정보 · 손실 기록 · 생성 샘플 json 저장"""
    res = {"device": dev, "V": V, "params": n_params, "config": vars(a), "ln_V": round(math.log(V), 3),
           "history": hist, "best_val": best, "sample_300": sample, "train_sec": train_sec, "gen_sec": gen_sec}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def viz_loss_curve(V, hist, out) -> Path:
    """train · val 손실 곡선 그림 저장"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps, trains, vals = [], [], []
    for h in hist:                                                  # 기록표 → 그래프용 열 3개
        steps.append(h["step"])
        trains.append(h["train"])
        vals.append(h["val"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, trains, label="train")
    ax.plot(steps, vals, label="val")
    ax.axhline(math.log(V), ls="--", c="gray", label="ln V (uniform)")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.legend()
    fig.tight_layout()
    png = out.with_name("ex32_loss.png")
    fig.savefig(png, dpi=120)
    return png


def log_saved(out, png) -> None:
    """저장 경로 출력"""
    print("저장:", out, png)


if __name__ == "__main__":
    main()
