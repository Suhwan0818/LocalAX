"""ex21_bpe_train_q.py — 바이트 BPE 토크나이저 직접 학습

- 대응 슬라이드: 실습 2 · '구현 확인 포인트'(76)
- 내용
  1) get_stats(인접 쌍 빈도) · merge(쌍 → 새 토큰) 구현
  2) 학습 루프: 바이트(0~255)에서 시작해 --merges 회 병합 → 병합 표(회차·쌍·빈도·새 토큰)
  3) encode·decode 왕복 검사 decode(encode(s)) == s
  4) 확인: "aaabdaaabac" 3회 병합 == [258, 100, 258, 97, 99]
  5) 라이브러리 대조: HF tokenizers BpeTrainer(어휘 256+merges=556) + ByteLevel 토큰 수 비교
- 실행
  uv run python ex21_bpe_train_q.py --corpus data/ex21_corpus_ko.txt --merges 300
  uv run python ex21_bpe_train_q.py --corpus data/ex21_corpus_en.txt --merges 300
- 산출물: results/ex21_bpe.json (말뭉치 이름별로 누적 저장 → ko·en 비교)
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
TEST_SENTENCES = {  # 같은 뜻의 한·영 문장 → 학습 후 토큰 수 비교
    "ko": "민지는 아침 여덟 시에 학교에 간다.",
    "en": "Minji goes to school at eight in the morning.",
}


def get_stats(ids: list[int]) -> Counter:
    # TODO(1): 인접한 두 토큰 쌍 (a, b)의 등장 빈도 세기
    #   힌트: 리스트와 한 칸 민 리스트를 나란히 묶으면 인접 쌍이 나옴
    #   힌트: 빈도 세기는 collections.Counter
    raise NotImplementedError("TODO(1)를 구현할 것")


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    # TODO(2): 왼쪽부터 훑으며 pair가 나오면 new_id 하나로 바꾸기 (겹침 없이)
    #   힌트: 인덱스 i를 직접 움직이는 while 루프
    #   힌트: 쌍이 맞으면 new_id를 넣고 i += 2, 아니면 원래 토큰을 넣고 i += 1
    raise NotImplementedError("TODO(2)를 구현할 것")


def train(text: str, n_merges: int) -> tuple[dict, list[dict]]:
    """반환: merges {(a,b): new_id}, 병합 표 rows"""
    ids = list(text.encode("utf-8"))
    merges: dict[tuple[int, int], int] = {}
    vocab = {i: bytes([i]) for i in range(256)}
    rows = []
    # TODO(3): n_merges 회 — 최빈 쌍 선택 → 새 id(256+k) 부여 → ids 갱신 → 표 기록
    #   힌트: get_stats(ids).most_common(1)[0] → (최빈 쌍, 빈도)
    #   힌트: 쌍이 없거나 빈도 < 2면 중단
    #   힌트: merges[pair] = new_id, vocab[new_id] = 두 토큰 바이트를 이어 붙인 것
    #   힌트: rows에 step·pair·freq·new_id·token(바이트를 utf-8 decode, errors='replace') 기록
    raise NotImplementedError("TODO(3)를 구현할 것")
    return merges, rows


def encode(text: str, merges: dict) -> list[int]:
    ids = list(text.encode("utf-8"))
    # TODO(4a): 학습 순서(new_id가 작은 것)대로 적용 가능한 쌍을 반복 병합
    #   힌트: 학습 때 먼저 만든 병합(작은 new_id)부터 적용해야 학습과 같은 결과
    #   힌트: 현재 ids의 쌍 중 merges에 있는 것만 후보, 후보가 없으면 종료
    #   힌트: min()의 key 인자로 '병합 순서'를 기준 삼기 — merges에 없는 쌍은 무한대 취급
    raise NotImplementedError("TODO(4a)를 구현할 것")
    return ids


def decode(ids: list[int], merges: dict) -> str:
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), n in sorted(merges.items(), key=lambda kv: kv[1]):
        vocab[n] = vocab[a] + vocab[b]
    # TODO(4b): id → 바이트 이어 붙이기 → utf-8 디코드
    #   힌트: vocab 값은 bytes — 문자열이 아니라 바이트로 먼저 이어 붙인 뒤 한 번에 디코드
    #   힌트: 한 토큰이 글자 중간에서 잘릴 수 있으므로 decode의 errors 인자 필요
    raise NotImplementedError("TODO(4b)를 구현할 것")


def sanity() -> list[int]:
    ids = list("aaabdaaabac".encode())
    for k in range(3):
        pair = get_stats(ids).most_common(1)[0][0]
        ids = merge(ids, pair, 256 + k)
    return ids


def hf_compare(corpus: Path, vocab_size: int, sentences: dict) -> dict:
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    except ImportError:
        return {"skipped": "tokenizers 미설치"}
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
    tk.train([str(corpus)], tr)
    return {"vocab_size": tk.get_vocab_size(),
            "test_tokens": {k: len(tk.encode(s).ids) for k, s in sentences.items()},
            "note": "ByteLevel은 공백 기준 사전 분할 → 단어 경계를 넘는 병합이 없음(직접 구현과 차이)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/ex21_corpus_ko.txt")
    ap.add_argument("--merges", type=int, default=300)
    ap.add_argument("--out", default="results/ex21_bpe.json")
    a = ap.parse_args()
    corpus = Path(a.corpus) if Path(a.corpus).exists() else HERE / a.corpus
    lines = corpus.read_text(encoding="utf-8").splitlines()
    text = "\n".join(l for l in lines if not l.startswith("※"))  # 머리말(출처 표기) 제외

    s = sanity()
    print("[확인] aaabdaaabac 3회 병합:", s, "OK" if s == [258, 100, 258, 97, 99] else "FAIL")
    merges, rows = train(text, a.merges)
    print(f"\n말뭉치 {corpus.name}: {len(text)}자 · {len(text.encode())}바이트 · 병합 {len(rows)}회")
    print(f"{'회차':>4} {'쌍':>12} {'빈도':>5} {'새 토큰':>7}  문자열")
    for r in rows[:20]:
        print(f"{r['step']:>4} {str(tuple(r['pair'])):>12} {r['freq']:>5} {r['new_id']:>7}  {r['token']!r}")
    test = {}
    for k, sent in TEST_SENTENCES.items():
        ids = encode(sent, merges)
        test[k] = {"sentence": sent, "bytes": len(sent.encode()), "tokens": len(ids),
                   "roundtrip_ok": decode(ids, merges) == sent}
    corpus_ids = encode(text, merges)
    rt_ok = decode(corpus_ids, merges) == text and all(v["roundtrip_ok"] for v in test.values())
    print("\n테스트 문장 토큰 수:", {k: v["tokens"] for k, v in test.items()}, "| 왕복 검사:", "통과" if rt_ok else "실패")
    print("읽는 법: 학습한 말뭉치 언어의 문장이 더 적은 토큰으로 압축됨 — 같은 코드라도 말뭉치가 병합 규칙을 정함")
    hf = hf_compare(corpus, 256 + a.merges, TEST_SENTENCES)
    print("HF BpeTrainer 대조:", hf)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    allres = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    allres["sanity_aaabdaaabac"] = s
    allres[corpus.stem] = {"corpus": corpus.name, "chars": len(text), "bytes": len(text.encode()),
                           "merges": len(rows), "corpus_tokens": len(corpus_ids),
                           "compression": round(len(text.encode()) / len(corpus_ids), 2),
                           "merge_table_top20": rows[:20], "test": test, "roundtrip_ok": rt_ok,
                           "hf_bpetrainer": hf}
    out.write_text(json.dumps(allres, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
