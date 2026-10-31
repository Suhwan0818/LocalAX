"""ex22_token_cost_sol.py — 모델별 한국어 토큰 수·비용 비교 (정답)

- 대응 슬라이드: M1 실습 3 · '같은 문서, 토크나이저 7종' / '실습 3 · 측정 · 산출물'
- 내용
  1) 로더 통일: HF AutoTokenizer(로컬 캐시) · tiktoken(cl100k_base·o200k_base)을 encode 함수 하나로
  2) 토큰 밀도: 토큰/1,000자 · 토큰/바이트
  3) 한/영 배율: ex21 병렬 문장(ko 토큰 ÷ en 토큰)
  4) 비용 계산기: --price-in/--price-out(USD/1M 토큰) → 요청 1만 건 비용
- 선행: pyc tools_fetch.py (Pod를 켤 때마다, 토크나이저 파일·샘플)
- 실행: pyc ex22_token_cost_sol.py --text data/tinystories_ko_sample.txt --n-lines 2000
- 산출물: results/ex22_tokens.json · results/ex22_tokens.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

HERE = Path(__file__).parent
HF_TOKENIZERS = [
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
    "skt/A.X-4.0-Light",
    "HuggingFaceTB/SmolLM2-135M",
    "Qwen/Qwen3-0.6B",
]
TIKTOKEN = ["cl100k_base", "o200k_base"]


def make_hf_encode(t):
    """HF 토크나이저 t → encode 함수(text -> list[int]) · 특수 토큰(BOS 등) 없이 본문 토큰만"""
    def encode(s: str) -> list[int]:
        return t.encode(s, add_special_tokens=False)
    return encode                               # 부를 때마다 새 encode → 토크나이저마다 자기 t를 기억


def load_encoders() -> dict:
    """이름 → (encode 함수, 어휘 크기). 받지 못한 토크나이저는 건너뜀"""
    enc = {}
    # TODO-BEGIN(1): AutoTokenizer와 tiktoken을 같은 모양(text -> list[int])으로 통일
    from transformers import AutoTokenizer
    for m in HF_TOKENIZERS:
        try:
            t = AutoTokenizer.from_pretrained(m, local_files_only=True)
            enc[m.split("/")[-1]] = (make_hf_encode(t), len(t))
        except Exception as e:
            print(f"[안내] {m} 건너뜀 ({type(e).__name__}) — 조치: pyc tools_fetch.py")
    try:
        import tiktoken
        for e in TIKTOKEN:
            t = tiktoken.get_encoding(e)
            enc[e] = (t.encode, t.n_vocab)
    except Exception as e:
        print(f"[안내] tiktoken 건너뜀 ({type(e).__name__})")
    # TODO-END
    return enc


def density(encode, text: str) -> dict:
    # TODO-BEGIN(2): 토큰 수, 토큰/1,000자, 토큰/바이트
    n = len(encode(text))
    return {"tokens": n, "tok_per_1k_chars": round(n / len(text) * 1000, 1),
            "tok_per_byte": round(n / len(text.encode("utf-8")), 3)}
    # TODO-END


def ko_en_ratio(encode, ko: list[str], en: list[str]) -> float:
    # TODO-BEGIN(3): 병렬 문장 전체의 ko 토큰 합 ÷ en 토큰 합
    ko_tokens = 0
    for s in ko:
        ko_tokens += len(encode(s))
    en_tokens = 0
    for s in en:
        en_tokens += len(encode(s))
    return round(ko_tokens / en_tokens, 2)
    # TODO-END


def cost_10k(tok_per_1k_chars: float, req_chars: int, out_chars: int, p_in: float, p_out: float) -> float:
    # TODO-BEGIN(4): 요청 1건 = 입력 req_chars자 + 출력 out_chars자(한국어). 1만 건 비용(USD)
    tin = tok_per_1k_chars * req_chars / 1000
    tout = tok_per_1k_chars * out_chars / 1000
    return round(10_000 * (tin * p_in + tout * p_out) / 1e6, 2)
    # TODO-END


def read_parallel(name: str) -> list[str]:
    lines = (HERE / "data" / name).read_text(encoding="utf-8").splitlines()
    sentences = []
    for line in lines:
        if line and not line.startswith("※"):   # 빈 줄·머리말(출처 표기) 제외
            sentences.append(line)
    return sentences


def by_density(r: dict) -> float:
    """정렬 기준: 토큰/1,000자 (작을수록 효율적)"""
    return r["tok_per_1k_chars"]


# ---------- wrapper 이름표 ----------
# 실제 동작: setup_ 준비 · run_ 실행 · save_ 저장
# 디버깅·로그·시각화: debug_ 검증 · log_ 화면 출력 · viz_ 그래프
def main() -> None:
    a = setup_args()
    p, text, ko, en = setup_data(a)
    rows = run_measure_tokenizers(a, text, ko, en) #TODO(2,3,4)
    debug_exit_if_no_rows(rows)
    run_sort_by_density(rows)
    log_report(a, p, text, rows)
    out = save_results(a, p, text, rows)
    png = viz_density_bars(rows, out)
    log_saved(out, png)


def setup_args() -> argparse.Namespace:
    """실행 인자 읽기"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="data/tinystories_ko_sample.txt")
    ap.add_argument("--n-lines", type=int, default=2000)
    ap.add_argument("--price-in", type=float, default=0.5, help="USD / 1M 입력 토큰 (가정값)")
    ap.add_argument("--price-out", type=float, default=2.0, help="USD / 1M 출력 토큰 (가정값, 보통 입력의 수배)")
    ap.add_argument("--req-chars", type=int, default=2000, help="요청 1건 입력 글자 수")
    ap.add_argument("--out-chars", type=int, default=500, help="요청 1건 출력 글자 수")
    ap.add_argument("--out", default="results/ex22_tokens.json")
    a = ap.parse_args()
    return a


def setup_data(a) -> tuple:
    """문서 앞 n_lines줄 · ex21 병렬 문장 읽기"""
    p = Path(a.text)
    if not p.exists():                          # 실행 위치에 없으면 스크립트 폴더 기준
        p = HERE / a.text
    with open(p, encoding="utf-8") as f:
        lines = []
        for line in f:
            if len(lines) >= a.n_lines:         # 앞 n_lines줄만 읽기
                break
            lines.append(line)
        text = "".join(lines).replace("<|endoftext|>", "\n")  # 이야기 구분자 제거
    ko, en = read_parallel("ex21_corpus_ko.txt"), read_parallel("ex21_corpus_en.txt")
    return p, text, ko, en


def run_measure_tokenizers(a, text, ko, en) -> list:
    """토크나이저별 토큰 밀도 · 한/영 배율 · 1만 건 비용 측정"""
    rows = []
    for name, (encode, V) in load_encoders().items():
        d = density(encode, text) #TODO(2)
        r = {"tokenizer": name, "vocab": V}
        r.update(d)                             # 밀도 3항목(tokens · tok_per_1k_chars · tok_per_byte) 이어 붙이기
        r["ko_en_ratio"] = ko_en_ratio(encode, ko, en) #TODO(3)
        r["cost_10k_usd"] = cost_10k(d["tok_per_1k_chars"], a.req_chars, a.out_chars, a.price_in, a.price_out) #TODO(4)
        rows.append(r)
    return rows


def debug_exit_if_no_rows(rows) -> None:
    """측정 결과가 비었으면 조치 안내 후 종료"""
    if not rows:
        raise SystemExit("[안내] 사용할 토크나이저 없음 — 조치: pyc tools_fetch.py")


def run_sort_by_density(rows) -> None:
    """측정 결과를 토큰 밀도 오름차순으로 정렬"""
    rows.sort(key=by_density)                   # 토큰/1k자 오름차순 → rows[0]이 가장 효율적


def log_report(a, p, text, rows) -> None:
    """토크나이저 비교표 · 8K 컨텍스트 문서 수 · 읽는 법 출력"""
    print(f"문서: {p.name} 앞 {a.n_lines}줄 · {len(text):,}자\n")
    print(f"{'토크나이저':<34}{'어휘 V':>9}{'토큰/1k자':>10}{'한/영':>7}{'1만건$':>9}")
    for r in rows:
        print(f"{r['tokenizer']:<34}{r['vocab']:>9,}{r['tok_per_1k_chars']:>10}{r['ko_en_ratio']:>7}{r['cost_10k_usd']:>9}")
    best = rows[0]["tok_per_1k_chars"]
    print("\n읽는 법: 배율 차이 = 같은 컨텍스트에 들어가는 문서 수 차이. 예) 8K 컨텍스트에 2,000자 문서 몇 건?")
    for r in rows:
        print(f"  {r['tokenizer']:<34} {int(8192 // (r['tok_per_1k_chars'] * 2)):>3}건 (최저 대비 {r['tok_per_1k_chars'] / best:.2f}배 토큰)")


def save_results(a, p, text, rows) -> Path:
    """측정 조건 · 결과 표 json 저장"""
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"text": p.name, "n_lines": a.n_lines, "chars": len(text),
                               "price_in": a.price_in, "price_out": a.price_out,
                               "req_chars": a.req_chars, "out_chars": a.out_chars, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def viz_density_bars(rows, out) -> Path:
    """토크나이저별 토큰 밀도 가로 막대 그림 저장"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    names = []
    densities = []
    for r in rows:
        names.append(r["tokenizer"])
        densities.append(r["tok_per_1k_chars"])
    ax.barh(names, densities)
    ax.set_xlabel("tokens per 1,000 Korean chars (lower = more efficient)")
    ax.invert_yaxis()
    fig.tight_layout()
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=120)
    return png


def log_saved(out, png) -> None:
    """저장 경로 출력"""
    print("저장:", out, png)


if __name__ == "__main__":
    main()
