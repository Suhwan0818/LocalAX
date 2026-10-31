"""ex31_attention_sol.py — Attention·인과 마스크 직접 구현 (정답)

- 대응 슬라이드: M2 'PyTorch 구현 ① attn() 5줄 · ② CausalMHA 생성자 · ③ forward' / M2 실습 1
- 내용
  TODO(1) attn(): 점수 · √hs 스케일 · tril 마스크 · softmax · 가중합 (수치 예 [0.25, 0.25, 0.50] 재현)
  TODO(2) SDPA 대조: torch.allclose(mine, sdpa, atol=1e-5)
  TODO(3) 누설 검사(x[:, 3:] 변경 시 y[:, :3] 불변)
  TODO(4) CausalMHA: view·transpose로 (B,H,T,hs) 분할 → 결합 → proj
  TODO(5) T 스윕(O(n²)): q에도 채워져 있음
  채워진 코드: √d_k 확인
- 실행: pyc ex31_attention_sol.py [--max-T 4096]
- 산출물: results/ex31_attention.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


def attn(q, k, v, causal: bool = True):                 # (..., T, hs)
    # TODO-BEGIN(1): s = q kᵀ / √hs → (causal이면) tril 밖 -inf → softmax(-1) @ v
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
    def __init__(self, C, H):
        super().__init__()
        assert C % H == 0
        self.H = H
        self.qkv, self.proj = nn.Linear(C, 3 * C), nn.Linear(C, C)

    def forward(self, x, causal: bool = True):           # (B, T, C)
        # TODO-BEGIN(4): 투영 → 분할 (B,H,T,hs) → attn → 결합 (B,T,C) → proj
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.H, -1).transpose(1, 2)    # (B, T, C) → (B, T, H, hs) → (B, H, T, hs)
        k = k.view(B, T, self.H, -1).transpose(1, 2)
        v = v.view(B, T, self.H, -1).transpose(1, 2)
        y = attn(q, k, v, causal).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)
        # TODO-END


def numeric_example() -> list[float]:
    """3번째 쿼리가 키 3개를 보는 경우: 점수 [0, 0, ln2] → softmax = [0.25, 0.25, 0.50] (hs=1, 구성된 예)"""
    q = torch.tensor([[0.0], [0.0], [1.0]])
    k = torch.tensor([[0.0], [0.0], [math.log(2)]])
    v = torch.eye(3)
    weights = attn(q, k, v)[2].tolist()          # v=I → 3번째 행 = 가중치 그대로
    out = []
    for w in weights:
        out.append(round(w, 2))
    return out


def sdpa_check(dev) -> dict:
    torch.manual_seed(0)
    q = torch.randn(2, 4, 16, 8, device=dev)
    k = torch.randn(2, 4, 16, 8, device=dev)
    v = torch.randn(2, 4, 16, 8, device=dev)
    # TODO-BEGIN(2): F.scaled_dot_product_attention(is_causal=True)와 allclose
    mine = attn(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return {"allclose": bool(torch.allclose(mine, ref, atol=1e-5)), "max_abs_diff": float((mine - ref).abs().max())}
    # TODO-END


@torch.no_grad()
def leak_check(dev, causal: bool = True) -> float:
    torch.manual_seed(0)
    m = CausalMHA(32, 4).to(dev)
    x = torch.randn(1, 8, 32, device=dev)
    # TODO-BEGIN(3): 미래 위치 x[:, 3:]만 바꿔도 y[:, :3]가 그대로인지 → 최대 변화량
    x2 = x.clone()
    x2[:, 3:] = torch.randn_like(x2[:, 3:])
    return float((m(x, causal)[:, :3] - m(x2, causal)[:, :3]).abs().max())
    # TODO-END


def sqrt_dk_check(d: int = 64, n: int = 10_000) -> float:
    a, b = torch.randn(n, d), torch.randn(n, d)
    return round(float((a * b).sum(-1).std()), 2)   # ≈ √64 = 8 → 그래서 √hs로 나눔


@torch.no_grad()
def t_sweep(dev, max_T: int, H: int = 8, hs: int = 64) -> list[dict]:
    rows, T = [], 256
    while T <= max_T:
        q = torch.randn(1, H, T, hs, device=dev)
        k = torch.randn(1, H, T, hs, device=dev)
        v = torch.randn(1, H, T, hs, device=dev)
        # TODO-BEGIN(5): attn 시간(워밍업 1회 + 3회 평균) · 점수 행렬 메모리(H·T²·4바이트)
        attn(q, k, v)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            attn(q, k, v)
        if dev == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 3 * 1000
        # TODO-END
        rows.append({"T": T, "ms": round(ms, 2), "score_MB": round(H * T * T * 4 / 2**20, 1)})
        T *= 2
    for i in range(len(rows) - 1):                      # 이웃한 두 T: rows[i] → rows[i + 1]
        rows[i + 1]["time_ratio_vs_prev"] = round(rows[i + 1]["ms"] / max(rows[i]["ms"], 1e-6), 2)
    return rows


# ---------- wrapper 이름표 ----------
# 실제 동작: setup_ 준비 · run_ 실행 · save_ 저장
# 디버깅·로그·시각화: debug_ 검증 · log_ 화면 출력 · viz_ 그래프
def main() -> None:
    a = setup_args()
    dev = setup_device(a)
    res = run_attention_checks(dev) #TODO(1,2,3,4)
    log_attention_checks(res)
    run_t_sweep(a, dev, res) #TODO(1)
    log_t_sweep(res)
    run_judge_pass(res)
    debug_print_pass(res)
    out = save_results(a, res)
    log_saved(out)
    debug_exit_code(res)


def setup_args() -> argparse.Namespace:
    """실행 인자 읽기"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-T", type=int, default=4096, help="CPU면 1024 정도로 줄일 것")
    default_dev = "cpu"
    if torch.cuda.is_available():               # GPU가 보이면 기본값 cuda
        default_dev = "cuda"
    ap.add_argument("--device", default=default_dev)
    ap.add_argument("--out", default="results/ex31_attention.json")
    a = ap.parse_args()
    return a


def setup_device(a) -> str:
    """--device 값을 실행 장치로 사용"""
    dev = a.device
    return dev


def run_attention_checks(dev) -> dict:
    """수치 예 · SDPA 대조 · 누설 검사 · √d_k 확인 → 결과 dict 시작"""
    res = {"device": dev, "numeric_example": numeric_example(), "sdpa": sdpa_check(dev), #TODO(1,2)
           "leak_max_change_causal": leak_check(dev, True), "leak_max_change_nomask": leak_check(dev, False), #TODO(1,3,4)
           "dot_std_d64": sqrt_dk_check()}
    return res


def log_attention_checks(res) -> None:
    """수치 예 · SDPA 대조 · 누설 검사 · √d_k 확인 결과 출력"""
    print("[1] 수치 예:", res["numeric_example"], "(기대 [0.25, 0.25, 0.5])")
    print("[2] SDPA 대조:", res["sdpa"])
    print(f"[3] 누설 검사: causal 변화 {res['leak_max_change_causal']:.2e} · 마스크 없음 변화 {res['leak_max_change_nomask']:.2e}")
    print(f"    √d_k 확인: randn(64) 내적 1만 회 표준편차 = {res['dot_std_d64']} (≈ 8)")


def run_t_sweep(a, dev, res) -> None:
    """T 스윕 → res에 기록"""
    res["t_sweep"] = t_sweep(dev, a.max_T) #TODO(1)


def log_t_sweep(res) -> None:
    """T 스윕 표 · 읽는 법 출력"""
    print(f"\n{'T':>6}{'ms':>10}{'점수MB':>10}{'시간배율':>9}")
    for r in res["t_sweep"]:
        print(f"{r['T']:>6}{r['ms']:>10}{r['score_MB']:>10}{r.get('time_ratio_vs_prev', ''):>9}")
    print("읽는 법: T 2배 → 점수 메모리 정확히 4배, 시간은 약 4배(작은 T에서는 고정 비용 때문에 덜 늘 수 있음)")


def run_judge_pass(res) -> None:
    """핵심 항목 통과 여부 판정 → res에 기록"""
    res["PASS"] = res["sdpa"]["allclose"] and res["leak_max_change_causal"] == 0.0 \
        and res["numeric_example"] == [0.25, 0.25, 0.5]


def debug_print_pass(res) -> None:
    """[CORE PASS] · [FAIL] 출력"""
    print("[CORE PASS] ex31" if res["PASS"] else "[FAIL] ex31 — allclose·누설 검사 확인")


def save_results(a, res) -> Path:
    """결과 json 저장"""
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def log_saved(out) -> None:
    """저장 경로 출력"""
    print("저장:", out)


def debug_exit_code(res) -> None:
    """통과면 0 · 실패면 1로 종료"""
    if res["PASS"]:                             # 종료 코드: 통과 0 · 실패 1
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
