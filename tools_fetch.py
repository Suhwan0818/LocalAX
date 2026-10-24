"""사전 준비: 실습 3(ex22)용 토크나이저 파일·tiktoken 인코딩 다운로드 (학생이 매 Pod에서 1회 실행).

HF 캐시(/workspace/hf)·tiktoken 캐시는 Pod 삭제 시 사라지므로 새 Pod마다 다시 실행.
한국어 샘플(data/tinystories_ko_sample.txt)은 이미 포함되어 있어 받지 않음.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")   # .env의 HF_TOKEN 등을 환경변수로 등록
os.environ.setdefault("HF_HOME", "/workspace/hf")   # README의 export를 빠뜨려도 같은 캐시 위치 사용
from huggingface_hub import hf_hub_download
import tiktoken
for m in ["LGAI-EXAONE/EXAONE-4.0-1.2B","kakaocorp/kanana-1.5-2.1b-instruct-2505","naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B","skt/A.X-4.0-Light","HuggingFaceTB/SmolLM2-135M"]:
    for f in ["tokenizer.json","tokenizer_config.json","special_tokens_map.json","config.json"]:   # config.json: AutoTokenizer가 토크나이저 종류 판별에 사용
        try: hf_hub_download(m,f)
        except Exception as e: print("skip",m,f,type(e).__name__)
for e in ["cl100k_base","o200k_base"]: tiktoken.get_encoding(e)
print("토크나이저 준비 완료")
