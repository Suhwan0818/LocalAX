# 01 LLM과 Transformer — 모델 · 토큰 · Attention

모델 로드와 다음 토큰 예측, 바이트 BPE 토크나이저, 모델별 한국어 토큰 비용, Attention·인과 마스크를 직접 다루는 실습 폴더입니다.
`_q.py` 파일은 `TODO`를 채워 완성하는 문제 파일입니다.

작업 폴더: `/workspace/w01` (본인 GitHub 저장소). 모든 명령은 이 폴더 안에서 실행.

## 0. 환경 준비 (매 Pod 1회)
```bash
grep -q HF_HOME ~/.bashrc || echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc
export HF_HOME=/workspace/hf
cp .env.example .env
uv venv --clear --system-site-packages --python /usr/bin/python3.12
uv sync --inexact
git config --global user.name "이름"
git config --global user.email "메일주소"
```

## 1. 모델·토크나이저 다운로드
```bash
uv run hf download Qwen/Qwen3-0.6B
uv run hf download Qwen/Qwen3.5-0.8B
uv run python tools_fetch.py
```

## 2. 사전 점검 (수정 없이 실행만)
```bash
mkdir -p results
uv run python ex00a_local_env_check_q.py --out results/ex00a_local_env_check.json   # uv·Git·.gitignore·.env
uv run python ex00b_gpu_env_check_q.py --out results/ex00b_gpu_env.json             # GPU·CUDA·torch
uv run python ex00c_model_data_check_q.py                                            # 모델 캐시·data 확인
```
FAIL·`[환경 조치 필요]`가 나오면 코드가 아니라 환경 문제. 코드는 고치지 말고 표의 조치 문구를 따를 것.

## 3. 실습 실행
```bash
uv run python ex01_model_load_q.py --out results/ex01.json
uv run python ex21_bpe_train_q.py --corpus data/ex21_corpus_ko.txt --merges 300
uv run python ex21_bpe_train_q.py --corpus data/ex21_corpus_en.txt --merges 300
uv run python ex22_token_cost_q.py --text data/tinystories_ko_sample.txt --n-lines 2000
uv run python ex31_attention_q.py                    # GPU 없으면 --device cpu --max-T 1024
```

| 파일 | 내용 | 산출물 |
|---|---|---|
| ex00a_local_env_check_q.py | uv·Python·Git·.gitignore·.env 점검 | results/ex00a_local_env_check.json |
| ex00b_gpu_env_check_q.py | GPU·CUDA·VRAM 점검, bf16 TFLOPS·대역폭 실측 | results/ex00b_gpu_env.json |
| ex00c_model_data_check_q.py | HF 캐시 모델·실습 데이터 확인 | (화면 출력) |
| ex01_model_load_q.py | Qwen3-0.6B vs Qwen3.5-0.8B config 비교, 다음 토큰 top-5, greedy 생성 | results/ex01.json |
| ex21_bpe_train_q.py | 바이트 BPE 직접 학습 · encode/decode 왕복 · HF tokenizers 대조 | results/ex21_bpe.json |
| ex22_token_cost_q.py | 토크나이저별 한국어 토큰 밀도 · 한/영 배율 · 비용 계산 | results/ex22_tokens.json · png |
| ex31_attention_q.py | attn() · CausalMHA 구현, SDPA 대조, 미래 누설 검사 | results/ex31_attention.json |
| tools_fetch.py | ex22용 토크나이저·tiktoken 인코딩 다운로드 | HF 캐시 |
