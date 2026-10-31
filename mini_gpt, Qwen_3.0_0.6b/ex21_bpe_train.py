"""ex21_bpe_train_sol.py — 바이트 BPE 토크나이저 직접 학습 (정답)

- 대응 슬라이드: M1 실습 2 · 'BPE 토크나이저 직접 학습' / '실습 2 · 측정 · 산출물'
- 내용
  1) get_stats(인접 쌍 빈도) · merge(쌍 → 새 토큰) 구현
  2) 학습 루프: 바이트(0~255)에서 시작해 --merges 회 병합 → 병합 표(회차·쌍·빈도·새 토큰)
  3) encode·decode 왕복 검사 decode(encode(s)) == s
  4) 확인: "aaabdaaabac" 3회 병합 == [258, 100, 258, 97, 99]
  5) 라이브러리 대조: HF tokenizers BpeTrainer(어휘 256+merges=556) + ByteLevel 토큰 수 비교
- 실행
  pyc ex21_bpe_train_sol.py --corpus data/ex21_corpus_ko.txt --merges 300
  pyc ex21_bpe_train_sol.py --corpus data/ex21_corpus_en.txt --merges 300
- 산출물: results/ex21_bpe.json (말뭉치 이름별로 누적 저장 → ko·en 비교)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

HERE = Path(__file__).parent
TEST_SENTENCES = {  # 학습 말뭉치에 없는 같은 뜻의 한·영 문장 → 학습 후 토큰 수 비교
    "ko": "수지는 오후 세 시에 도서관에서 책을 읽는다.",
    "en": "Suji reads a book at the library at three in the afternoon.",
}


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    # TODO-BEGIN(1): 인접한 두 토큰 쌍 (a, b)의 등장 빈도 세기
    stats = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        stats[pair] = stats.get(pair, 0) + 1
    return stats
    # TODO-END


def most_frequent(stats: dict[tuple[int, int], int]) -> tuple[tuple[int, int] | None, int]:
    """최빈 쌍과 빈도 · 동률이면 먼저 센 쌍 (minbpe와 같은 규칙) · 빈 dict면 (None, 0)"""
    best, best_freq = None, 0
    for pair, freq in stats.items():
        if freq > best_freq:          # '>'라서 동률이면 앞의 쌍 유지
            best, best_freq = pair, freq
    return best, best_freq


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    # TODO-BEGIN(2): 왼쪽부터 훑으며 pair가 나오면 new_id 하나로 바꾸기 (겹침 없이)
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
    # TODO-END


def train(text: str, n_merges: int) -> tuple[dict, list[dict]]:
    """반환: merges {(a,b): new_id}, 병합 표 rows"""
    ids = list(text.encode("utf-8"))
    merges: dict[tuple[int, int], int] = {}
    vocab = {}
    for i in range(256):                        # 처음 어휘 = 바이트 0~255
        vocab[i] = bytes([i])
    rows = []
    # TODO-BEGIN(3): n_merges 회 — 최빈 쌍 선택 → 새 id(256+k) 부여 → ids 갱신 → 표 기록
    for k in range(n_merges):
        stats = get_stats(ids)                  # ① 쌍 세기
        best, freq = most_frequent(stats)       # ② 최빈 쌍
        if best is None or freq < 2:
            break
        new_id = 256 + k
        ids = merge(ids, best, new_id)          # ③ 병합
        merges[best] = new_id                   # ④ 갱신: 규칙·어휘 기록
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        rows.append({"step": k + 1, "pair": list(best), "freq": freq, "new_id": new_id,
                     "token": vocab[new_id].decode("utf-8", errors="replace")})
    # TODO-END
    return merges, rows


def encode(text: str, merges: dict) -> list[int]:
    ids = list(text.encode("utf-8"))
    # TODO-BEGIN(4a): 병합 규칙을 학습 순서대로 하나씩 적용 (빈도 계산 없음)
    for pair, new_id in merges.items():         # dict는 넣은 순서 유지 = 학습 순서 merge_1 → merge_K
        ids = merge(ids, pair, new_id)
    # TODO-END
    return ids


def decode(ids: list[int], merges: dict) -> str:
    vocab = {}
    for i in range(256):                        # 처음 어휘 = 바이트 0~255
        vocab[i] = bytes([i])
    for (a, b), n in merges.items():            # 학습 순서 → a·b가 먼저 vocab에 있음
        vocab[n] = vocab[a] + vocab[b]
    # TODO-BEGIN(4b): id → 바이트 이어 붙이기 → utf-8 디코드
    parts = []
    for i in ids:
        parts.append(vocab[i])
    return b"".join(parts).decode("utf-8", errors="replace")
    # TODO-END


def sanity() -> list[int]:
    ids = list("aaabdaaabac".encode())
    for k in range(3):
        pair, _ = most_frequent(get_stats(ids))
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
    test_tokens = {}
    for k, s in sentences.items():
        test_tokens[k] = len(tk.encode(s).ids)
    return {"vocab_size": tk.get_vocab_size(),
            "test_tokens": test_tokens,
            "note": "ByteLevel은 공백 기준 사전 분할 → 단어 경계를 넘는 병합이 없음(직접 구현과 차이)"}


# ---------- wrapper 이름표 ----------
# 실제 동작: setup_ 준비 · run_ 실행 · save_ 저장
# 디버깅·로그·시각화: debug_ 검증 · log_ 화면 출력 · viz_ 그래프
def main() -> None:
    a = setup_args()
    corpus, text = setup_corpus(a)
    s = run_sanity_example() #TODO(1,2)
    debug_sanity_example(s)
    merges, rows = run_train(a, text) #TODO(1,2,3)
    log_merge_table(corpus, text, rows)
    test, corpus_ids, rt_ok = run_roundtrip(text, merges) #TODO(2,4a,4b)
    log_roundtrip(test, rt_ok)
    hf = run_hf_compare(a, corpus)
    log_hf_compare(hf)
    out = save_results(a, s, corpus, text, rows, corpus_ids, test, rt_ok, hf)
    log_saved(out)


def setup_args() -> argparse.Namespace:
    """실행 인자 읽기"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/ex21_corpus_ko.txt")
    ap.add_argument("--merges", type=int, default=300)
    ap.add_argument("--out", default="results/ex21_bpe.json")
    a = ap.parse_args()
    return a


def setup_corpus(a) -> tuple:
    """말뭉치 읽기 → 머리말 뺀 본문"""
    corpus = Path(a.corpus)
    if not corpus.exists():                     # 실행 위치에 없으면 스크립트 폴더 기준
        corpus = HERE / a.corpus
    lines = corpus.read_text(encoding="utf-8").splitlines()
    body = []
    for line in lines:
        if not line.startswith("※"):            # 머리말(출처 표기) 제외
            body.append(line)
    text = "\n".join(body)
    return corpus, text


def run_sanity_example() -> list:
    """손계산 예(aaabdaaabac) 3회 병합"""
    s = sanity() #TODO(1,2)
    return s


def debug_sanity_example(s) -> None:
    """병합 결과를 손계산 값과 대조 → OK · FAIL 출력"""
    print("[확인] aaabdaaabac 3회 병합:", s, "OK" if s == [258, 100, 258, 97, 99] else "FAIL")


def run_train(a, text) -> tuple:
    """BPE 학습 → 병합 규칙 · 병합 표"""
    merges, rows = train(text, a.merges) #TODO(1,2,3)
    return merges, rows


def log_merge_table(corpus, text, rows) -> None:
    """말뭉치 크기 · 병합 표 앞 20행 출력"""
    print(f"\n말뭉치 {corpus.name}: {len(text)}자 · {len(text.encode())}바이트 · 병합 {len(rows)}회")
    print(f"{'회차':>4} {'쌍':>12} {'빈도':>5} {'새 토큰':>7}  문자열")
    for r in rows[:20]:
        print(f"{r['step']:>4} {str(tuple(r['pair'])):>12} {r['freq']:>5} {r['new_id']:>7}  {r['token']!r}")


def run_roundtrip(text, merges) -> tuple:
    """테스트 문장 · 말뭉치 전체 encode → decode 왕복 검사"""
    test = {}
    for k, sent in TEST_SENTENCES.items():
        ids = encode(sent, merges) #TODO(2,4a)
        test[k] = {"sentence": sent, "bytes": len(sent.encode()), "tokens": len(ids),
                   "roundtrip_ok": decode(ids, merges) == sent} #TODO(4b)
    corpus_ids = encode(text, merges) #TODO(2,4a)
    rt_ok = decode(corpus_ids, merges) == text #TODO(4b)
    for v in test.values():                     # 테스트 문장 하나라도 실패면 실패
        if not v["roundtrip_ok"]:
            rt_ok = False
    return test, corpus_ids, rt_ok


def log_roundtrip(test, rt_ok) -> None:
    """테스트 문장 토큰 수 · 왕복 검사 결과 · 읽는 법 출력"""
    tokens = {}
    for k, v in test.items():
        tokens[k] = v["tokens"]
    print("\n테스트 문장 토큰 수:", tokens, "| 왕복 검사:", "통과" if rt_ok else "실패")
    print("읽는 법: 학습한 말뭉치 언어의 문장이 더 적은 토큰으로 압축됨 — 같은 코드라도 말뭉치가 병합 규칙을 정함")


def run_hf_compare(a, corpus) -> dict:
    """HF BpeTrainer로 같은 말뭉치 학습 → 토큰 수 대조"""
    hf = hf_compare(corpus, 256 + a.merges, TEST_SENTENCES)
    return hf


def log_hf_compare(hf) -> None:
    """HF 대조 결과 출력"""
    print("HF BpeTrainer 대조:", hf)


def save_results(a, s, corpus, text, rows, corpus_ids, test, rt_ok, hf) -> Path:
    """말뭉치 이름별로 json에 누적 저장"""
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    allres = {}
    if out.exists():                            # 이전 실행 결과가 있으면 이어서 누적
        allres = json.loads(out.read_text(encoding="utf-8"))
    allres["sanity_aaabdaaabac"] = s
    allres[corpus.stem] = {"corpus": corpus.name, "chars": len(text), "bytes": len(text.encode()),
                           "merges": len(rows), "corpus_tokens": len(corpus_ids),
                           "compression": round(len(text.encode()) / len(corpus_ids), 2),
                           "merge_table_top20": rows[:20], "test": test, "roundtrip_ok": rt_ok,
                           "hf_bpetrainer": hf}
    out.write_text(json.dumps(allres, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def log_saved(out) -> None:
    """저장 경로 출력"""
    print("저장:", out)


if __name__ == "__main__":
    main()
