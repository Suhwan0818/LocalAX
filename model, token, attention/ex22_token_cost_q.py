"""ex22_token_cost_q.py — 모델별 한국어 토큰 수·비용 비교

- 대응 슬라이드: 실습 3 · '토큰 비용 구조와 절감 습관'(85)
- 내용
  1) 로더 통일: HF AutoTokenizer(로컬 캐시) · tiktoken(cl100k_base·o200k_base)을 encode 함수 하나로
  2) 토큰 밀도: 토큰/1,000자 · 토큰/바이트
  3) 한/영 배율: ex21 병렬 문장(ko 토큰 ÷ en 토큰)
  4) 비용 계산기: --price-in/--price-out(USD/1M 토큰) → 요청 1만 건 비용
- 선행: uv run python tools_fetch.py (매 Pod 1회, 토크나이저·tiktoken 캐시)
- 실행: uv run python ex22_token_cost_q.py --text data/tinystories_ko_sample.txt --n-lines 2000
- 산출물: results/ex22_tokens.json · results/ex22_tokens.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")   # .env의 HF_TOKEN 등을 환경변수로 등록

HERE = Path(__file__).parent
HF_TOKENIZERS = [
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "kakaocorp/kanana-1.5-2.1b-instruct-2505",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
    "skt/A.X-4.0-Light",
    "HuggingFaceTB/SmolLM2-135M",
    "Qwen/Qwen3-0.6B",
]
TIKTOKEN = ["cl100k_base", "o200k_base"]


def load_encoders() -> dict:
    """이름 → (encode 함수, 어휘 크기). 받지 못한 토크나이저는 건너뜀"""
    enc = {}
    # 제공 코드(1, 수정 불필요): AutoTokenizer와 tiktoken을 같은 모양(text -> list[int])으로 통일
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    for m in HF_TOKENIZERS:
        try:
            try:
                t = AutoTokenizer.from_pretrained(m, local_files_only=True)
            except Exception:   # config 검증 실패 모델(예: kanana)은 tokenizer.json만으로 로드
                t = PreTrainedTokenizerFast.from_pretrained(m, local_files_only=True)
            enc[m.split("/")[-1]] = (lambda s, t=t: t.encode(s, add_special_tokens=False), len(t))
        except Exception as e:
            print(f"[안내] {m} 건너뜀 ({type(e).__name__}) — 조치: uv run python tools_fetch.py")
    try:
        import tiktoken
        for e in TIKTOKEN:
            t = tiktoken.get_encoding(e)
            enc[e] = (t.encode, t.n_vocab)
    except Exception as e:
        print(f"[안내] tiktoken 건너뜀 ({type(e).__name__})")
    # /제공 코드
    return enc


def density(encode, text: str) -> dict:
    # TODO(2): 토큰 수, 토큰/1,000자, 토큰/바이트
    #   힌트: 토큰 수 = len(encode(text))
    #   힌트: 글자 수는 len(text), 바이트 수는 len(text.encode('utf-8'))
    raise NotImplementedError("TODO(2)를 구현할 것")


def ko_en_ratio(encode, ko: list[str], en: list[str]) -> float:
    # TODO(3): 병렬 문장 전체의 ko 토큰 합 ÷ en 토큰 합
    #   힌트: 문장 하나씩 비율을 평균내지 말고, 언어별 토큰 수를 먼저 전부 합산
    raise NotImplementedError("TODO(3)를 구현할 것")


def cost_10k(tok_per_1k_chars: float, req_chars: int, out_chars: int, p_in: float, p_out: float) -> float:
    # TODO(4): 요청 1건 = 입력 req_chars자 + 출력 out_chars자(한국어). 1만 건 비용(USD)
    #   힌트: 토큰 수 ≈ tok_per_1k_chars × 글자 수 / 1000 (입력·출력 따로)
    #   힌트: 가격은 USD / 1M 토큰 → (입력토큰 × p_in + 출력토큰 × p_out) / 1e6
    #   힌트: 1만 건이므로 × 10_000
    raise NotImplementedError("TODO(4)를 구현할 것")


def read_parallel(name: str) -> list[str]:
    return [l for l in (HERE / "data" / name).read_text(encoding="utf-8").splitlines() if l and not l.startswith("※")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="data/tinystories_ko_sample.txt")
    ap.add_argument("--n-lines", type=int, default=2000)
    ap.add_argument("--price-in", type=float, default=0.5, help="USD / 1M 입력 토큰 (가정값)")
    ap.add_argument("--price-out", type=float, default=2.0, help="USD / 1M 출력 토큰 (가정값, 보통 입력의 수배)")
    ap.add_argument("--req-chars", type=int, default=2000, help="요청 1건 입력 글자 수")
    ap.add_argument("--out-chars", type=int, default=500, help="요청 1건 출력 글자 수")
    ap.add_argument("--out", default="results/ex22_tokens.json")
    a = ap.parse_args()
    p = Path(a.text) if Path(a.text).exists() else HERE / a.text
    with open(p, encoding="utf-8") as f:
        text = "".join(l for _, l in zip(range(a.n_lines), f)).replace("<|endoftext|>", "\n")  # 이야기 구분자 제거
    ko, en = read_parallel("ex21_corpus_ko.txt"), read_parallel("ex21_corpus_en.txt")

    rows = []
    for name, (encode, V) in load_encoders().items():
        d = density(encode, text)
        r = {"tokenizer": name, "vocab": V, **d, "ko_en_ratio": ko_en_ratio(encode, ko, en)}
        r["cost_10k_usd"] = cost_10k(d["tok_per_1k_chars"], a.req_chars, a.out_chars, a.price_in, a.price_out)
        rows.append(r)
    if not rows:
        raise SystemExit("[안내] 사용할 토크나이저 없음 — 조치: uv run python tools_fetch.py")
    rows.sort(key=lambda r: r["tok_per_1k_chars"])
    print(f"문서: {p.name} 앞 {a.n_lines}줄 · {len(text):,}자\n")
    print(f"{'토크나이저':<34}{'어휘 V':>9}{'토큰/1k자':>10}{'한/영':>7}{'1만건$':>9}")
    for r in rows:
        print(f"{r['tokenizer']:<34}{r['vocab']:>9,}{r['tok_per_1k_chars']:>10}{r['ko_en_ratio']:>7}{r['cost_10k_usd']:>9}")
    best = rows[0]["tok_per_1k_chars"]
    print("\n읽는 법: 배율 차이 = 같은 컨텍스트에 들어가는 문서 수 차이. 예) 8K 컨텍스트에 2,000자 문서 몇 건?")
    for r in rows:
        print(f"  {r['tokenizer']:<34} {int(8192 // (r['tok_per_1k_chars'] * 2)):>3}건 (최저 대비 {r['tok_per_1k_chars'] / best:.2f}배 토큰)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"text": p.name, "n_lines": a.n_lines, "chars": len(text),
                               "price_in": a.price_in, "price_out": a.price_out,
                               "req_chars": a.req_chars, "out_chars": a.out_chars, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh([r["tokenizer"] for r in rows], [r["tok_per_1k_chars"] for r in rows])
    ax.set_xlabel("tokens per 1,000 Korean chars (lower = more efficient)")
    ax.invert_yaxis(); fig.tight_layout()
    png = out.with_suffix(".png"); fig.savefig(png, dpi=120)
    print("저장:", out, png)


if __name__ == "__main__":
    main()
