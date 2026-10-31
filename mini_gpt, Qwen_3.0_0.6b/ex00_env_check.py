"""ex00_env_check.py — 1회차 환경 점검

- 대응 슬라이드: M1 [실습1] RunPod 환경 점검과 모델 로드 (1/2)
- 점검: Python·torch·CUDA·GPU·VRAM·HF 캐시 모델·실습 데이터
- 실행: pyc ex00_env_check.py
"""
from __future__ import annotations
import json, os, platform, sys
from pathlib import Path

HERE = Path(__file__).parent
REQUIRED_MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"]
REQUIRED_DATA = ["data/tinystories_ko_sample.txt", "data/ex21_corpus_ko.txt", "data/ex21_corpus_en.txt"]


def hf_cache_dir() -> Path:
    # 우선순위: HF_HUB_CACHE > HF_HOME/hub (RunPod: HF_HOME=/root/hf) > ~/.cache/huggingface/hub
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


def check() -> dict:
    r: dict = {"python": platform.python_version(), "os": platform.platform()}
    ok = True
    # torch 버전, CUDA 사용 가능 여부, GPU 이름·VRAM(GB)·compute capability를 r에 기록
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
    cache = hf_cache_dir()
    hf_models = {}
    for m in REQUIRED_MODELS:                   # 캐시 폴더 이름 = models--조직--모델
        found = (cache / ("models--" + m.replace("/", "--"))).exists()
        hf_models[m] = found
        if not found:                           # 하나라도 없으면 실패
            ok = False
    r["hf_models"] = hf_models
    data = {}
    for d in REQUIRED_DATA:                     # 실습 데이터 (스크립트 폴더 기준 경로)
        found = (HERE / d).exists()
        data[d] = found
        if not found:                           # 하나라도 없으면 실패
            ok = False
    r["data"] = data
    r["PASS"] = ok
    return r


# ---------- wrapper 이름표 ----------
# 실제 동작: setup_ 준비 · run_ 실행 · save_ 저장
# 디버깅·로그·시각화: debug_ 검증 · log_ 화면 출력 · viz_ 그래프
def main() -> None:
    res = run_env_check()
    log_report(res)
    debug_exit_code(res)


def run_env_check() -> dict:
    """환경 점검 → 항목별 결과 · 통과 여부"""
    res = check()
    return res


def log_report(res) -> None:
    """점검 결과를 json 형식으로 출력"""
    print(json.dumps(res, ensure_ascii=False, indent=2))


def debug_exit_code(res) -> None:
    """통과면 0 · 실패면 1로 종료"""
    exit_code = 1
    if res["PASS"]:                             # 통과면 0 (셸에서 성공)
        exit_code = 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
