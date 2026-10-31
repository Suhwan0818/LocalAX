"""ex01_model_load_sol.py — 모델 로드와 다음 토큰 예측 관찰 (정답)

- 대응 슬라이드: M1 [실습1] 모델 로드 · '다음 토큰 예측과 디코더 전용 구조'
- 내용
  1) 모델 config에서 층 수·hidden·vocab·head·layer_types 읽기 (Qwen3-0.6B vs Qwen3.5-0.8B)
  2) 프롬프트 한 개의 다음 토큰 확률 top-5 출력 (softmax(logits[-1]))
  3) greedy로 한 토큰씩 직접 생성 루프 → model.generate 결과와 일치 확인
- 실행: pyc ex01_model_load_sol.py [--model Qwen/Qwen3-0.6B] [--out results/ex01.json]
"""
from __future__ import annotations
import argparse, json, time
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPT = "대한민국의 수도는"


def describe(model_id: str) -> dict:
    cfg = AutoConfig.from_pretrained(model_id)
    t = getattr(cfg, "text_config", cfg)
    lt = getattr(t, "layer_types", None)
    layer_types = {}
    if lt:
        for kind in lt:                         # 층마다 어텐션 종류 세기 (full · linear 등)
            layer_types[kind] = layer_types.get(kind, 0) + 1
    else:
        layer_types["full_attention"] = t.num_hidden_layers   # 목록 없음 = 전 층 full
    return {
        "model": model_id, "layers": t.num_hidden_layers, "hidden": t.hidden_size,
        "vocab": t.vocab_size, "q_heads": t.num_attention_heads, "kv_heads": t.num_key_value_heads,
        "head_dim": getattr(t, "head_dim", t.hidden_size // t.num_attention_heads),
        "tie_word_embeddings": bool(getattr(t, "tie_word_embeddings", getattr(cfg, "tie_word_embeddings", False))),
        "layer_types": layer_types,
    }


@torch.no_grad()
def next_token_topk(model, tok, prompt: str, k: int = 5) -> list[tuple[str, float]]:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    # TODO-BEGIN(1): 마지막 위치의 logits → softmax → top-k (토큰 문자열, 확률) 목록 반환
    logits = model(ids).logits[0, -1].float()
    z = logits - logits.max()                   # 최댓값 빼기 (exp 넘침 방지 · 확률은 그대로)
    exp_z = torch.exp(z)                        # ① exp
    probs = exp_z / exp_z.sum()                 # ② 합 → ③ 합으로 나누기 (확률 합 1)
    top_probs, top_ids = probs.topk(k)          # 확률 큰 순 k개 (확률, 어휘 id)
    top = []
    for rank in range(k):                       # rank 0 = 1위
        token = tok.decode([int(top_ids[rank])])
        top.append((token, round(float(top_probs[rank]), 4)))
    return top
    # TODO-END


@torch.no_grad()
def greedy_loop(model, tok, prompt: str, n: int = 12) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    # TODO-BEGIN(2): n번 반복 — 전체 시퀀스를 넣고 argmax 토큰을 이어 붙이기 (KV 캐시 없이)
    for _ in range(n):
        nxt = model(ids).logits[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    # TODO-END
    return tok.decode(ids[0], skip_special_tokens=True)


# ---------- wrapper 이름표 ----------
# 실제 동작: setup_ 준비 · run_ 실행 · save_ 저장
# 디버깅·로그·시각화: debug_ 검증 · log_ 화면 출력 · viz_ 그래프
def main() -> None:
    a = setup_args()
    out = run_describe_configs(a)
    dev = setup_device()
    tok, model = setup_model_timed(a, dev, out)
    run_model_stats(dev, model, out)
    run_next_token_top5(tok, model, out) #TODO(1)
    run_generate_compare(dev, tok, model, out) #TODO(2)
    log_report(out)
    save_results(a, out)


def setup_args() -> argparse.Namespace:
    """실행 인자 읽기"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--compare", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    return a


def run_describe_configs(a) -> dict:
    """두 모델의 config 요약 → 결과 dict 시작"""
    out = {"configs": [describe(a.model), describe(a.compare)]}
    return out


def setup_device() -> str:
    """GPU가 있으면 GPU 사용"""
    dev = "cpu"
    if torch.cuda.is_available():               # GPU가 있으면 GPU에서 실행
        dev = "cuda"
    return dev


def setup_model_timed(a, dev, out) -> tuple:
    """토크나이저 · 모델 로드 → 로드 시간을 out에 기록"""
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    out["load_sec"] = round(time.time() - t0, 1)
    return tok, model


def run_model_stats(dev, model, out) -> None:
    """가중치 수 · VRAM 사용량 측정 → out에 기록"""
    n_params = 0
    for p in model.parameters():                # 가중치 텐서마다 원소 수 더하기 (묶인 가중치는 한 번)
        n_params += p.numel()
    out["params_M"] = round(n_params / 1e6, 1)
    out["vram_GB"] = 0
    if dev == "cuda":                           # CPU면 0 그대로
        out["vram_GB"] = round(torch.cuda.memory_allocated() / 1024**3, 2)


def run_next_token_top5(tok, model, out) -> None:
    """프롬프트의 다음 토큰 확률 top-5 → out에 기록"""
    out["prompt"] = PROMPT
    out["top5"] = next_token_topk(model, tok, PROMPT) #TODO(1)


def run_generate_compare(dev, tok, model, out) -> None:
    """직접 만든 greedy 생성 · model.generate 결과 대조 → out에 기록"""
    mine = greedy_loop(model, tok, PROMPT) #TODO(2)
    ref = tok.decode(model.generate(**tok(PROMPT, return_tensors="pt").to(dev), max_new_tokens=12, do_sample=False)[0], skip_special_tokens=True)
    out["greedy_loop"], out["generate"], out["match"] = mine, ref, mine == ref


def log_report(out) -> None:
    """결과 전체를 json 형식으로 출력"""
    print(json.dumps(out, ensure_ascii=False, indent=2))


def save_results(a, out) -> None:
    """--out을 준 경우에만 json 저장"""
    if a.out:
        open(a.out, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
