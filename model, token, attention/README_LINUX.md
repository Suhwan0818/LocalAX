# 01 LLM과 Transformer — RunPod 실행 준비

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
