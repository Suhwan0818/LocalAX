"""ex33_mask_ablation_sol.py — 인과 마스크 제거 비교 (정답)

- 대응 슬라이드: M2 실습 3 '마스크 제거 시 정답 누설' · '예상 결과와 해석' · '세 설정을 한 표에'
- 통제 실험: 실습 2(ex32) 모델에서 마스크만 끄고 나머지(데이터·seed·설정·스텝)는 동일
  설정 3종: causal=True(정상) · causal=False(마스크 없음) · y=x(안 밀림, 흔한 실수)
- 내용
  TODO(1) 같은 seed로 설정별 학습
  TODO(2) attn(q, k, v, causal: bool) 인자화
  TODO(3) 누설 검사: x[:, 3:] 변경 시 logits[:, :3] 변화량
  TODO(4) 같은 접두로 300자 생성
  TODO(5) 표·그래프 저장
- 기호: B: 배치 크기(문장 수) · T: 토큰 수(시퀀스 길이 · 최대 --T) · C: 임베딩 차원 · H: 헤드 수 · hs: 헤드 하나의 차원(C/H)
        V: 어휘 크기(글자 종류 수) · L: 블록 수(층 수) · d_ff: FFN 중간 차원 · (..., T, hs): 앞쪽 축은 무엇이든(B·H 등) 마지막 두 축만
        A: 어텐션 가중치(softmax 결과 · 행 합 1) · shift: 정답 y를 한 칸 미는지(y=x 실험이면 False) · ln V: 균등 추측 손실(V개 중 찍기)
        x[:, 3:]: 위치 3부터(미래) 입력 · logits[:, :3]: 위치 0~2(과거) 출력
- 실행: pyc ex33_mask_ablation_sol.py --steps 1500 --out results/ex33.json
- 산출물: results/ex33.json · results/ex33_compare.png
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
SETTINGS = [  # (이름, causal, shift)
    ("causal=True", True, True),
    ("causal=False", False, True),
    ("y=x (no shift)", True, False),
]


def attn(q, k, v, causal: bool = True):                 # (..., T, hs)
    # TODO-BEGIN(2): causal=False면 마스크를 건너뜀
    T, hs = q.shape[-2:]
    s = q @ k.transpose(-2, -1) / math.sqrt(hs)          # ① 점수 · ② 스케일
    if causal:
        m = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        s = s.masked_fill(~m, float("-inf"))               # 미래 칸 -inf (softmax 전)
    # ③ softmax(행마다): 행 최댓값 빼기 → exp → 행 합으로 나누기
    e = torch.exp(s - s.amax(dim=-1, keepdim=True))       # 최댓값 빼도 결과 같음 · exp 넘침 방지 · exp(-inf) = 0
    a = e / e.sum(dim=-1, keepdim=True)                   # 가중치 A · 행 합 1
    return a @ v                                          # ④ 가중합
    # TODO-END


class CausalMHA(nn.Module):
    def __init__(self, C, H, causal):
        super().__init__()
        self.H, self.causal = H, causal
        self.qkv, self.proj = nn.Linear(C, 3 * C), nn.Linear(C, C)

    def forward(self, x):                                # (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.H, -1).transpose(1, 2)    # (B, T, C) → (B, T, H, hs) → (B, H, T, hs)
        k = k.view(B, T, self.H, -1).transpose(1, 2)
        v = v.view(B, T, self.H, -1).transpose(1, 2)
        y = attn(q, k, v, self.causal).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, C, d_ff):
        super().__init__()
        self.w1, self.w3, self.w2 = nn.Linear(C, d_ff, bias=False), nn.Linear(C, d_ff, bias=False), nn.Linear(d_ff, C, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, C, H, d_ff, causal):
        super().__init__()
        self.ln1, self.attn, self.ln2, self.ffn = nn.RMSNorm(C), CausalMHA(C, H, causal), nn.RMSNorm(C), SwiGLU(C, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, V, T, C, H, L, d_ff, causal=True):
        super().__init__()
        self.T = T
        self.wte, self.wpe = nn.Embedding(V, C), nn.Embedding(T, C)
        blocks = []
        for _ in range(L):                                           # Block L개 차례로 쌓기
            blocks.append(Block(C, H, d_ff, causal))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f, self.lm_head = nn.RMSNorm(C), nn.Linear(C, V, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):                            # (B, T)
        pos = torch.arange(idx.size(1), device=idx.device)          # 위치 0, 1, …, T-1
        x = self.wte(idx) + self.wpe(pos)                            # (B, T, C)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))                          # (B, T, V)
        loss = None                                                  # 생성 때는 정답 없음 → loss 없음
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())   # (B·T, V) · (B·T,)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n):
        self.eval()
        for _ in range(n):
            logits, _ = self(idx[:, -self.T:])                       # 최근 T개만 (wpe 범위 안)
            probs = logits[:, -1].softmax(-1)                        # (B, T, V) → 마지막 위치 (B, V)
            nxt = torch.multinomial(probs, 1)                        # 토큰 1개 샘플링 (B, 1)
            idx = torch.cat([idx, nxt], dim=1)                       # idx 뒤에 붙이기
        self.train()
        return idx


def get_batch(data, B, T, dev, shift=True):
    ix = torch.randint(len(data) - T - 1, (B,))                     # 시작점 i를 B개 무작위로
    if shift:
        o = 1                                                       # 정상: 한 칸 밀린 정답
    else:
        o = 0                                                       # y=x 실험: 밀지 않음
    xs, ys = [], []
    for i in ix:
        xs.append(data[i:i + T])                                    # 입력 (T,)
        ys.append(data[i + o:i + o + T])                            # 정답 (T,) · o칸 밀림
    x = torch.stack(xs)                                             # (B, T)
    y = torch.stack(ys)
    return x.to(dev), y.to(dev)


@torch.no_grad()
def estimate(model, data, B, T, dev, shift, iters=20):
    model.eval()
    total = 0.0
    for _ in range(iters):                                          # 배치 iters개 손실 평균
        _, loss = model(*get_batch(data, B, T, dev, shift))
        total += loss.item()
    model.train()
    return total / iters


@torch.no_grad()
def leak_check(model, data, T, dev) -> float:
    # TODO-BEGIN(3): 같은 입력에서 x[:, 3:]만 바꾸고 logits[:, :3]의 최대 변화량
    model.eval()
    x, _ = get_batch(data, 1, T, dev)                               # 입력 하나 (1, T)
    x2 = x.clone()
    x2[:, 3:] = torch.randint_like(x2[:, 3:], int(model.wte.num_embeddings))   # 위치 3부터(미래)만 무작위 글자로
    logits, _ = model(x)                                            # (1, T, V)
    logits2, _ = model(x2)
    d = float((logits[:, :3] - logits2[:, :3]).abs().max())         # 위치 0~2(과거) 출력의 최대 변화량
    model.train()
    return d
    # TODO-END


# ---------- wrapper 이름표 ----------
# 실제 동작: setup_ 준비 · run_ 실행 · save_ 저장
# 디버깅·로그·시각화: debug_ 검증 · log_ 화면 출력 · viz_ 그래프
def main() -> None:
    a = setup_args()
    dev = setup_device()
    V, stoi, chars, train_d, val_d, prefix = setup_data(a)
    results = run_train_all_settings(a, dev, V, stoi, chars, train_d, val_d, prefix) #TODO(1,2,3,4)
    log_compare_table(results)
    out, png = save_and_viz_results(a, dev, V, results) #TODO(5)
    log_saved(out, png)


def setup_args() -> argparse.Namespace:
    """실행 인자 읽기"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-chars", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="results/ex33.json")
    a = ap.parse_args()
    return a


def setup_device() -> str:
    """GPU가 있으면 GPU 사용"""
    dev = "cpu"
    if torch.cuda.is_available():
        dev = "cuda"
    return dev


def setup_data(a) -> tuple:
    """글자 어휘 · 토큰 ID 열 · 학습/검증 분할 · 생성 접두"""
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
    prefix = text[:10]
    return V, stoi, chars, train_d, val_d, prefix


def run_train_all_settings(a, dev, V, stoi, chars, train_d, val_d, prefix) -> list:
    """설정 3종을 차례로 학습 · 누설 검사 · 생성"""
    results = []
    for name, causal, shift in SETTINGS:
        model, hist, t0 = run_train_one_setting(a, dev, V, train_d, val_d, causal, shift) #TODO(1,2)
        leak = leak_check(model, val_d, a.T, dev) #TODO(2,3)
        sample = run_generate(a, dev, stoi, chars, prefix, model) #TODO(2,4)
        r = {"setting": name, "causal": causal, "shift": shift, "history": hist,
             "train_final": hist[-1]["train"], "val_final": hist[-1]["val"],
             "leak_max_change": leak, "leak_pass": leak < 1e-5, "sample_300": sample,
             "sec": round(time.time() - t0, 1)}
        results.append(r)
        log_setting_result(name, leak, sample, r)
    return results


def run_train_one_setting(a, dev, V, train_d, val_d, causal, shift) -> tuple:
    """설정 하나 학습 → 모델 · 손실 기록 · 시작 시각"""
    # TODO-BEGIN(1): 설정마다 같은 seed로 모델·배치 순서 고정 후 학습
    torch.manual_seed(a.seed)
    model = GPT(V, a.T, a.C, a.H, a.L, a.d_ff, causal).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    hist, t0 = [], time.time()
    for step in range(a.steps + 1):
        if step % 100 == 0 or step == a.steps:
            tr = estimate(model, train_d, a.batch, a.T, dev, shift)    # train 먼저, val 다음
            va = estimate(model, val_d, a.batch, a.T, dev, shift)
            hist.append({"step": step, "train": round(tr, 4), "val": round(va, 4)})
        if step == a.steps:
            break
        _, loss = model(*get_batch(train_d, a.batch, a.T, dev, shift))
        opt.zero_grad(set_to_none=True)                         # 기울기 초기화
        loss.backward()                                         # 역전파
        opt.step()                                              # 가중치 갱신
    # TODO-END
    return model, hist, t0


def run_generate(a, dev, stoi, chars, prefix, model) -> str:
    """같은 접두로 300자 생성"""
    # TODO-BEGIN(4): 같은 접두로 300자 생성 (생성 시 seed 고정)
    torch.manual_seed(a.seed)
    prefix_ids = []
    for c in prefix:
        prefix_ids.append(stoi[c])
    idx = torch.tensor([prefix_ids], device=dev)                # (1, 접두 글자 수)
    gen_ids = model.generate(idx, 300)[0].tolist()              # 접두 + 생성 300개 id
    parts = []
    for i in gen_ids:                                           # id → 글자
        parts.append(chars[i])
    sample = "".join(parts)
    # TODO-END
    return sample


def log_setting_result(name, leak, sample, r) -> None:
    """설정 하나의 손실 · 누설 · 생성 앞부분 출력"""
    print(f"\n[{name}] train {r['train_final']:.3f} · val {r['val_final']:.3f} · 누설 변화 {leak:.2e} "
          f"({'통과' if r['leak_pass'] else '실패'}) · {r['sec']}s\n  생성: {sample[:120]!r}")


def log_compare_table(results) -> None:
    """설정 3종 비교표 · 읽는 법 출력"""
    print(f"\n{'설정':<18}{'train':>8}{'val':>8}{'누설':>6}")
    for r in results:
        print(f"{r['setting']:<18}{r['train_final']:>8.3f}{r['val_final']:>8.3f}{'통과' if r['leak_pass'] else '실패':>6}")
    print("읽는 법: causal=False의 낮은 loss = 오른쪽 토큰을 베낀 결과(측정 조건 ≠ 생성 조건) → 생성은 붕괴."
          " y=x는 누설 검사는 통과해도 항등 함수 학습. train loss가 유독 낮으면 정답 누설부터 의심")


def save_and_viz_results(a, dev, V, results) -> tuple:
    """결과 json 저장 + 세 곡선 그림 저장"""
    # TODO-BEGIN(5): json + 세 곡선(train 실선·val 점선)을 한 그림에
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": dev, "V": V, "config": vars(a), "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in results:
        steps, trains, vals = [], [], []
        for h in r["history"]:                                      # 기록표 → 그래프용 열 3개
            steps.append(h["step"])
            trains.append(h["train"])
            vals.append(h["val"])
        line, = ax.plot(steps, trains, label=f"{r['setting']} train")                     # train 실선
        ax.plot(steps, vals, ls="--", c=line.get_color(), label=f"{r['setting']} val")    # val 같은 색 점선
    ax.axhline(math.log(V), ls=":", c="gray", label="ln V")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = out.with_name("ex33_compare.png")
    fig.savefig(png, dpi=120)
    # TODO-END
    return out, png


def log_saved(out, png) -> None:
    """저장 경로 출력"""
    print("저장:", out, png)


if __name__ == "__main__":
    main()
