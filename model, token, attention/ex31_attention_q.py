"""ex31_attention_q.py — Attention·인과 마스크 직접 구현

- 대응 슬라이드: 실습 4 · 'PyTorch 구현 ① attn() 5줄 · ② CausalMHA 생성자 · ③ CausalMHA.forward'(140~142)
- 내용
  TODO(1) attn(): 점수 · √hs 스케일 · tril 마스크 · softmax · 가중합 (수치 예 [0.25, 0.25, 0.50] 재현)
  TODO(2) CausalMHA: view·transpose로 (B,H,T,hs) 분할 → 결합 → proj
  TODO(3) SDPA 대조: torch.allclose(mine, sdpa, atol=1e-5)
  TODO(4) 누설 검사(x[:, 3:] 변경 시 y[:, :3] 불변) — √d_k 확인·T 스윕(O(n²))은 제공 코드, 결과만 해석
- 실행: uv run python ex31_attention_q.py [--max-T 4096]   (GPU 없으면 --device cpu --max-T 1024)
- 산출물: results/ex31_attention.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


def attn(q, k, v, causal: bool = True):                 # (..., T, hs)
    # TODO(1): s = q kᵀ / √hs → (causal이면) tril 밖 -inf → softmax(-1) @ v
    #   힌트: q·k의 마지막 두 축이 (T, hs) — 행렬곱 전에 k의 마지막 두 축을 바꿔야 함 (.transpose)
    #   힌트: 하삼각 bool 행렬은 torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device)) — device 빠지면 GPU에서 오류
    #   힌트: 가릴 칸 채우기는 Tensor.masked_fill — 가릴 칸은 마스크의 반대(~)
    #   힌트: -inf는 softmax 후 정확히 0이 됨 → 미래 토큰 가중치 0
    raise NotImplementedError("TODO(1)를 구현할 것")


class CausalMHA(nn.Module):
    def __init__(self, C, H):
        super().__init__(); assert C % H == 0
        self.H = H
        self.qkv, self.proj = nn.Linear(C, 3 * C), nn.Linear(C, C)

    def forward(self, x, causal: bool = True):           # (B, T, C)
        # TODO(2): 투영 → 분할 (B,H,T,hs) → attn → 결합 (B,T,C) → proj
        #   힌트: self.qkv 출력의 마지막 축에 q·k·v가 C씩 붙어 있음 → Tensor.split
        #   힌트: 헤드 분할은 view로 C를 (H, hs)로 쪼갠 뒤 transpose로 H 축을 T 앞으로
        #   힌트: 결합은 분할의 역순 — transpose 뒤 view 대신 reshape를 쓰는 이유(메모리 연속성) 생각해 볼 것
        raise NotImplementedError("TODO(2)를 구현할 것")


def numeric_example() -> list[float]:
    """3번째 쿼리가 키 3개를 보는 경우: 점수 [0, 0, ln2] → softmax = [0.25, 0.25, 0.50] (hs=1, 구성된 예)"""
    q = torch.tensor([[0.0], [0.0], [1.0]])
    k = torch.tensor([[0.0], [0.0], [math.log(2)]])
    v = torch.eye(3)
    return [round(x, 2) for x in attn(q, k, v)[2].tolist()]  # v=I → 가중치 그대로


def sdpa_check(dev) -> dict:
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8, device=dev) for _ in range(3))
    # TODO(3): F.scaled_dot_product_attention(is_causal=True)와 allclose
    #   힌트: PyTorch 내장 어텐션: F.scaled_dot_product_attention — 인과 마스크 인자 확인
    #   힌트: 두 텐서 비교는 torch.allclose(atol), 최대 오차도 함께 반환
    raise NotImplementedError("TODO(3)를 구현할 것")


@torch.no_grad()
def leak_check(dev, causal: bool = True) -> float:
    torch.manual_seed(0)
    m = CausalMHA(32, 4).to(dev)
    x = torch.randn(1, 8, 32, device=dev)
    # TODO(4a): 미래 위치 x[:, 3:]만 바꿔도 y[:, :3]가 그대로인지 → 최대 변화량
    #   힌트: 원본을 덮어쓰지 않도록 복사(clone)한 뒤 앞 3칸 이후만 난수로 교체
    #   힌트: 앞 3칸 출력끼리 비교 — causal이면 0, 마스크가 없으면 0보다 큼
    raise NotImplementedError("TODO(4a)를 구현할 것")


def sqrt_dk_check(d: int = 64, n: int = 10_000) -> float:
    a, b = torch.randn(n, d), torch.randn(n, d)
    return round(float((a * b).sum(-1).std()), 2)   # ≈ √64 = 8 → 그래서 √hs로 나눔


@torch.no_grad()
def t_sweep(dev, max_T: int, H: int = 8, hs: int = 64) -> list[dict]:
    rows, T = [], 256
    while T <= max_T:
        q, k, v = (torch.randn(1, H, T, hs, device=dev) for _ in range(3))
        # 제공 코드(4b, 수정 불필요): attn 시간(워밍업 1회 + 3회 평균) · 점수 행렬 메모리(H·T²·4바이트)
        attn(q, k, v)
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3): attn(q, k, v)
        if dev == "cuda": torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 3 * 1000
        # /제공 코드
        rows.append({"T": T, "ms": round(ms, 2), "score_MB": round(H * T * T * 4 / 2**20, 1)})
        T *= 2
    for a, b in zip(rows, rows[1:]):
        b["time_ratio_vs_prev"] = round(b["ms"] / max(a["ms"], 1e-6), 2)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-T", type=int, default=4096, help="CPU면 1024 정도로 줄일 것")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/ex31_attention.json")
    a = ap.parse_args()
    dev = a.device
    res = {"device": dev, "numeric_example": numeric_example(), "sdpa": sdpa_check(dev),
           "leak_max_change_causal": leak_check(dev, True), "leak_max_change_nomask": leak_check(dev, False),
           "dot_std_d64": sqrt_dk_check()}
    print("[1] 수치 예:", res["numeric_example"], "(기대 [0.25, 0.25, 0.5])")
    print("[3] SDPA 대조:", res["sdpa"])
    print(f"[4] 누설 검사: causal 변화 {res['leak_max_change_causal']:.2e} · 마스크 없음 변화 {res['leak_max_change_nomask']:.2e}")
    print(f"    √d_k 확인: randn(64) 내적 1만 회 표준편차 = {res['dot_std_d64']} (≈ 8)")
    res["t_sweep"] = t_sweep(dev, a.max_T)
    print(f"\n{'T':>6}{'ms':>10}{'점수MB':>10}{'시간배율':>9}")
    for r in res["t_sweep"]:
        print(f"{r['T']:>6}{r['ms']:>10}{r['score_MB']:>10}{r.get('time_ratio_vs_prev', ''):>9}")
    print("읽는 법: T 2배 → 점수 메모리 정확히 4배, 시간은 약 4배(작은 T에서는 고정 비용 때문에 덜 늘 수 있음)")
    res["PASS"] = res["sdpa"]["allclose"] and res["leak_max_change_causal"] == 0.0 \
        and res["numeric_example"] == [0.25, 0.25, 0.5]
    print("[CORE PASS] ex31" if res["PASS"] else "[FAIL] ex31 — [1] 수치 예 [0.25,0.25,0.5] · [3] SDPA allclose · [4] causal 누설=0 중 실패 항목 확인")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", out)
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
