# LocalAX

LLM과 Transformer의 동작 원리를 직접 구현하며 익히는 실습 저장소입니다.
모델 로드 → 토크나이저 → Attention → 미니 GPT 학습 순서로 진행하며, 실행 환경은 RunPod GPU Pod(Ubuntu · CUDA 12.8+)입니다.

## 폴더 구성

| 폴더 | 주제 | 주요 실습 |
|---|---|---|
| [`model, token, attention/`](./model,%20token,%20attention/) | 모델 · 토큰 · Attention | 환경 점검(ex00a~c) · 모델 로드(ex01) · BPE 학습(ex21) · 토큰 비용(ex22) · Attention 구현(ex31) |
| [`mini_gpt, Qwen_3.0_0.6b/`](./mini_gpt,%20Qwen_3.0_0.6b/) | 미니 GPT · Qwen3-0.6B | 모델 로드 · BPE · 토큰 비용 · Attention · 미니 GPT 학습(ex32) · 인과 마스크 제거 실험(ex33) |

각 폴더는 독립된 uv 프로젝트(`pyproject.toml` · `uv.lock`)이며, 실행 방법은 폴더 안의 `README.md`에 있습니다.

## 실습 흐름

| 번호 | 실습 | 내용 |
|---|---|---|
| ex00 | 환경 점검 | Python · GPU · CUDA · HF 캐시 모델 · 데이터 확인 |
| ex01 | 모델 로드 | Qwen3-0.6B vs Qwen3.5-0.8B 구조 비교, 다음 토큰 확률 top-5, greedy 생성 |
| ex21 | BPE 토크나이저 | 바이트 BPE 병합 학습, encode/decode 왕복, HF tokenizers 대조 |
| ex22 | 토큰 비용 | 토크나이저별 한국어 토큰 밀도 · 한/영 배율 · 요청 비용 계산 |
| ex31 | Attention | scaled dot-product · 인과 마스크 · Multi-Head 구현, SDPA 대조 |
| ex32 | 미니 GPT | 임베딩 · 블록(잔차 · SwiGLU) · LM head 조립 후 한국어 문자 단위 학습 |
| ex33 | 마스크 제거 실험 | causal / non-causal / y=x 세 설정 비교로 정답 누설 확인 |

## 빠른 시작

```bash
cd "model, token, attention"            # 또는 "mini_gpt, Qwen_3.0_0.6b"
uv venv --system-site-packages --python /usr/bin/python3.12
uv sync --inexact
uv run hf download Qwen/Qwen3-0.6B
uv run python tools_fetch.py
```

`.env`는 `.env.example`을 복사해 만들고(`HF_TOKEN` 권장), Git에는 올리지 않습니다.
`.venv/` · `__pycache__/` · `models/` · `.env`는 `.gitignore`로 모든 하위 폴더에서 제외됩니다.
