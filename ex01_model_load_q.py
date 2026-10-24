"""ex01_model_load_q.py — 모델 로드와 다음 토큰 예측 관찰

- 대응 슬라이드: 실습 1 · '베이스·인스트럭트 비유와 실습'(20) · '로짓을 읽는 네 가지 포인트'(94)
- 내용
  1) 모델 config에서 층 수·hidden·vocab·head·layer_types 읽기 (Qwen3-0.6B vs Qwen3.5-0.8B)
  2) 프롬프트 한 개의 다음 토큰 확률 top-5 출력 (softmax(logits[-1]))
  3) greedy로 한 토큰씩 직접 생성 루프 → model.generate 결과와 일치 확인
- 실행: uv run python ex01_model_load_q.py [--model Qwen/Qwen3-0.6B] [--out results/ex01.json]
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")   # .env의 HF_TOKEN 등을 환경변수로 등록

PROMPT = "대한민국의 수도는"


def describe(model_id: str) -> dict:
    cfg = AutoConfig.from_pretrained(model_id)
    t = getattr(cfg, "text_config", cfg)
    lt = getattr(t, "layer_types", None)
    return {
        "model": model_id, "layers": t.num_hidden_layers, "hidden": t.hidden_size,
        "vocab": t.vocab_size, "q_heads": t.num_attention_heads, "kv_heads": t.num_key_value_heads,
        "head_dim": getattr(t, "head_dim", t.hidden_size // t.num_attention_heads),
        "tie_word_embeddings": bool(getattr(t, "tie_word_embeddings", getattr(cfg, "tie_word_embeddings", False))),
        "layer_types": {k: lt.count(k) for k in set(lt)} if lt else {"full_attention": t.num_hidden_layers},
    }


@torch.no_grad()
def next_token_topk(model, tok, prompt: str, k: int = 5) -> list[tuple[str, float]]:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    # TODO(1): 마지막 위치의 logits → softmax → top-k (토큰 문자열, 확률) 목록 반환
    #   힌트: model(ids).logits 모양은 (1, T, V) → 마지막 위치는 [0, -1]
    #   힌트: torch.softmax(..., dim=-1) 후 .topk(k)는 (값, 인덱스)를 돌려줌
    #   힌트: 토큰 문자열은 tok.decode([id])
    raise NotImplementedError("TODO(1)를 구현할 것")


@torch.no_grad()
def greedy_loop(model, tok, prompt: str, n: int = 12) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    # TODO(2): n번 반복 — 전체 시퀀스를 넣고 argmax 토큰을 이어 붙이기 (KV 캐시 없이)
    #   힌트: 매 반복마다 지금까지의 ids 전체를 model에 넣음 (캐시 없음)
    #   힌트: 다음 토큰 = 마지막 위치 logits의 .argmax()
    #   힌트: torch.cat([ids, 새토큰.view(1, 1)], dim=1)로 이어 붙임
    raise NotImplementedError("TODO(2)를 구현할 것")
    return tok.decode(ids[0], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--compare", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = {"configs": [describe(a.model), describe(a.compare)]}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    out["load_sec"] = round(time.time() - t0, 1)
    out["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    out["vram_GB"] = round(torch.cuda.memory_allocated() / 1024**3, 2) if dev == "cuda" else 0
    out["prompt"] = PROMPT
    out["top5"] = next_token_topk(model, tok, PROMPT)
    mine = greedy_loop(model, tok, PROMPT)
    ref = tok.decode(model.generate(**tok(PROMPT, return_tensors="pt").to(dev), max_new_tokens=12, do_sample=False)[0], skip_special_tokens=True)
    out["greedy_loop"], out["generate"], out["match"] = mine, ref, mine == ref
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
