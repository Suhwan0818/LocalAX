"""ex00c_model_data_check_q.py — 1회차 환경 점검

- 대응 슬라이드: 실습 1 사전 점검
- 점검: Python·torch·CUDA·GPU·VRAM·HF 캐시 모델·실습 데이터
- 실행: uv run python ex00c_model_data_check_q.py   (수정 없이 실행만 하는 사전 점검)
"""
from __future__ import annotations
import json, os, platform, sys
from pathlib import Path

HERE = Path(__file__).parent
REQUIRED_MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"]
REQUIRED_DATA = ["data/tinystories_ko_sample.txt", "data/ex21_corpus_ko.txt", "data/ex21_corpus_en.txt"]


def hf_cache_dir() -> Path:
    # 우선순위: HF_HUB_CACHE > HF_HOME/hub (RunPod: HF_HOME=/workspace/hf) > ~/.cache/huggingface/hub
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


def check() -> dict:
    r: dict = {"python": platform.python_version(), "os": platform.platform()}
    ok = True
    # 제공 코드(1, 수정 불필요): torch 버전, CUDA 사용 가능 여부, GPU 이름·VRAM(GB)·compute capability를 r에 기록
    import torch
    r["torch"] = torch.__version__
    r["cuda_available"] = torch.cuda.is_available()
    if r["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        r["gpu"] = p.name
        r["vram_gb"] = round(p.total_memory / 1024**3, 1)
        r["capability"] = f"sm_{p.major}{p.minor}"
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        r["bf16_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
    else:
        ok = False
    # /제공 코드
    cache = hf_cache_dir()
    r["hf_models"] = {m: (cache / ("models--" + m.replace("/", "--"))).exists() for m in REQUIRED_MODELS}
    r["data"] = {d: (HERE / d).exists() for d in REQUIRED_DATA}
    ok = ok and all(r["hf_models"].values()) and all(r["data"].values())
    r["PASS"] = ok
    return r


if __name__ == "__main__":
    res = check()
    print(json.dumps(res, ensure_ascii=False, indent=2))
    sys.exit(0 if res["PASS"] else 1)
