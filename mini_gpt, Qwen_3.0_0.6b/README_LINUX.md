# 01 LLM과 Transformer — Linux(RunPod 개인 Pod) 실행 안내

대상: RunPod 개인 Pod 24GB 이상 GPU, Ubuntu, 드라이버 CUDA 12.8+, 작업 폴더 `/workspace` (글로벌 볼륨 · 코드·결과가 남음). venv·모델은 `/root`에 두며 Pod를 끄면 사라짐(5절).

## 1. 준비
교재 zip을 `/workspace/ax_courses`에 푼 상태에서 실행(`00_사전_환경구성_복사용.md` 2-4). Pod를 새로 켤 때마다 다시 실행.
```bash
bash /workspace/ax_courses/_shared/pod_setup_uv_linux.sh --no-llama   # venv·pyc·HF_HOME 구성
```
패키지 목록은 이 폴더의 `pyproject.toml`·`uv.lock`을 씀(지우거나 고치지 말 것). `[실패]`가 없으면 설치 내용을 터미널에 적용하고 이동.
```bash
source ~/.bashrc                 # 터미널을 새로 열 때는 자동 적용
cd /workspace/ax_courses/week1   # 영문 경로(비ASCII 경로 금지)
echo $HF_HOME                    # /root/hf 가 나와야 함
```

## 2. 모델·데이터 사전 다운로드
글로벌 볼륨은 파일 잠금·rename을 지원하지 않아 모델을 `/root/hf`에 받음. Pod를 끄면 사라지므로 새로 켤 때마다 다시 실행.
```bash
hf download Qwen/Qwen3-0.6B
hf download Qwen/Qwen3.5-0.8B
pyc tools_fetch.py   # 토크나이저 파일·TinyStories-Korean 샘플
```

## 3. 실행 순서
```bash
pyc ex00_env_check.py            # HF_HUB_CACHE > HF_HOME/hub > ~/.cache 순으로 모델 확인
mkdir -p results
pyc ex01_model_load_q.py --out results/ex01.json
# M1 실습 2·3 (CPU로 충분)
pyc ex21_bpe_train_q.py --corpus data/ex21_corpus_ko.txt --merges 300
pyc ex21_bpe_train_q.py --corpus data/ex21_corpus_en.txt --merges 300
pyc ex22_token_cost_q.py --text data/tinystories_ko_sample.txt --n-lines 2000
# M2 실습 1·2·3 (GPU 권장)
pyc ex31_attention_q.py                       # CPU면 --device cpu --max-T 1024
pyc ex32_mini_gpt_q.py --steps 1500 --out results/ex32.json
pyc ex33_mask_ablation_q.py --steps 1500 --out results/ex33.json
```

| 순서 | 파일 | 대응 슬라이드 | 산출물 |
|---|---|---|---|
| 0 | ex00_env_check.py | M1 실습 1 환경 점검 | (화면 출력) |
| 1 | ex01_model_load_q.py | M1 실습 1 모델 로드 | results/ex01.json |
| 2 | ex21_bpe_train_q.py | M1 실습 2 BPE 토크나이저 직접 학습 | results/ex21_bpe.json (ko·en 누적) |
| 3 | ex22_token_cost_q.py | M1 실습 3 모델별 한국어 토큰 수·비용 | results/ex22_tokens.json · png |
| 4 | ex31_attention_q.py | M2 실습 1 Attention·인과 마스크 직접 구현 | results/ex31_attention.json |
| 5 | ex32_mini_gpt_q.py | M2 실습 2 미니 GPT 조립·학습 | results/ex32.json · ex32_loss.png |
| 6 | ex33_mask_ablation_q.py | M2 실습 3 인과 마스크 제거 비교 | results/ex33.json · ex33_compare.png |

- ex22는 `tools_fetch.py`로 받은 토크나이저만 사용(없는 것은 `[안내]` 후 건너뜀)
- ex33은 causal=True · causal=False · y=x(안 밀림) 세 설정을 같은 seed로 학습해 한 그림에 겹침

## 4. 24GB 이상 참고
- 0.6B~0.8B bf16 모델은 2GB 미만 → 24GB 이상에서 여유.

## 5. 수업 종료 · 다음 수업
- 코드·결과: `/workspace/ax_courses`(글로벌 볼륨)에 남음 · 별도 저장 불필요
- venv·`pyc`·모델: Pod를 끄면 사라짐 → 다음 수업은 1·2절부터 다시 실행
- 교재 zip은 다시 풀지 않음(고친 `_q.py`를 덮어씀)
- 내 PC로 받기: VS Code 탐색기에서 파일 우클릭 → Download
