"""사전 준비: 토크나이저 파일·한국어 TinyStories valid 다운로드 (Pod를 켤 때마다 실행)."""
from huggingface_hub import hf_hub_download
import tiktoken, pathlib
for m in ["LGAI-EXAONE/EXAONE-4.0-1.2B","naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B","skt/A.X-4.0-Light","HuggingFaceTB/SmolLM2-135M"]:
    for f in ["tokenizer.json","tokenizer_config.json","special_tokens_map.json","config.json"]:  # config.json: transformers 5 AutoTokenizer가 요구
        try: hf_hub_download(m,f)
        except Exception as e: print("skip",m,f,type(e).__name__)
for e in ["cl100k_base","o200k_base"]: tiktoken.get_encoding(e)
p=hf_hub_download("g0ster/TinyStories-Korean","TinyStories-Korean-valid.txt",repo_type="dataset")
txt=open(p,encoding="utf-8").read()
out=pathlib.Path(__file__).parent/"data"/"tinystories_ko_sample.txt"
out.write_text(txt[:8_000_000//3],encoding="utf-8")
print("chars",len(txt),"->",out, out.stat().st_size)
