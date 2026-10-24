"""
ex00b_gpu_env_check_q.py — GPU·CUDA·VRAM 검증 게이트

학습 목표
  - nvidia-smi 5필드(이름·드라이버·총 VRAM·사용 VRAM·compute_cap) 수집
  - CUDA 3종 버전 대조: 드라이버가 지원하는 CUDA ≥ torch가 빌드된 CUDA, nvcc는 선택
  - torch가 이 GPU 아키텍처(sm_86/sm_89/sm_90/sm_120 등)용 커널을 갖고 있는지 확인
  - bf16 matmul TFLOPS·메모리 복사 대역폭을 짧게 실측해 기준표와 비교
실행(수정 없이 실행만 하는 사전 점검):
  uv run python ex00b_gpu_env_check_q.py --out results/ex00b_gpu_env.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 콘솔 인코딩 통일(Linux는 기본 UTF-8)

HERE = Path(__file__).resolve().parent
FIELDS = "name,driver_version,memory.total,memory.used,compute_cap"
MIN_FREE_GB = 20
# 기준표: (총 VRAM MiB, bf16 텐서코어 dense TFLOPS 참고치, 메모리 대역폭 GB/s 이론치)
# RunPod 개인 Pod 24GB 기준(참고치). GeForce는 FP32 누산 bf16이 절반 속도.
REFERENCE = {
    "RTX 4090 24GB (RunPod 개인 Pod)": (24564, 82.6, 1008),
    "L4 24GB (RunPod 개인 Pod)": (23034, 121.0, 300),
    "RTX A5000 24GB (RunPod 개인 Pod)": (24564, 55.6, 768),
}


def sh(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                           encoding="utf-8", errors="replace")
        return p.returncode, p.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 127, ""


def query_nvidia_smi() -> dict:
    # --- 제공 코드(1, 수정 불필요): nvidia-smi --query-gpu=FIELDS --format=csv,noheader,nounits 결과를 dict로 ---
    code, out = sh(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"])
    if code != 0 or not out:
        return {"status": "FAIL", "detail": "nvidia-smi 실패 — GPU 미할당 또는 드라이버 문제"}
    vals = [v.strip() for v in out.splitlines()[0].split(",")]
    info = dict(zip(FIELDS.split(","), vals))
    info["memory.total"] = int(float(info["memory.total"]))
    info["memory.used"] = int(float(info["memory.used"]))
    _, head = sh(["nvidia-smi"])  # 드라이버 R61x부터 헤더 표기가 'CUDA UMD Version'으로 바뀜
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", head)
    info["driver_cuda"] = m.group(1) if m else ""
    warn = info["memory.used"] > 2048
    return {"status": "WARN" if warn else "OK", **info,
            "detail": f"{info['name']} / 드라이버 {info['driver_version']} (CUDA {info['driver_cuda']}) / "
                      f"{info['memory.used']}/{info['memory.total']} MiB 사용 / cc {info['compute_cap']}"
                      + (" — 다른 프로세스가 VRAM 2GB 이상 점유" if warn else "")}
    # --- /제공 코드 ---


def check_nvcc() -> dict:
    code, out = sh(["nvcc", "--version"])
    if code != 0:
        return {"status": "WARN", "detail": "nvcc 없음 — torch 휠은 자체 CUDA 런타임 포함이라 필수 아님(JIT 빌드 시만 필요)"}
    m = re.search(r"release ([\d.]+)", out)
    return {"status": "OK", "detail": f"nvcc {m.group(1) if m else '?'}"}


def ver_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", s)[:2])


def check_torch(smi: dict) -> dict:
    # --- 제공 코드(2, 수정 불필요): torch.version.cuda, is_available, capability, arch_list에 sm_XY 포함 여부, 드라이버 CUDA ≥ torch CUDA ---
    import torch
    avail = torch.cuda.is_available()
    if not avail:
        return {"status": "FAIL", "detail": f"torch {torch.__version__} CUDA 미사용 — CPU 휠 설치 여부 확인(cu128+ 인덱스)"}
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    # sm_86/sm_89/sm_90/sm_120 등: 정확히 일치하거나 같은 major의 하위 minor 커널이 있으면 바이너리 호환(OK)
    archs = [x for x in torch.cuda.get_arch_list() if x.startswith("sm_")]
    arch_ok = sm in archs or any(x[3:-1] == sm[3:-1] and int(x[-1]) <= int(sm[-1]) for x in archs)
    drv_ok = ver_tuple(smi.get("driver_cuda", "0")) >= ver_tuple(torch.version.cuda or "0")
    ok = arch_ok and drv_ok
    return {"status": "OK" if ok else "FAIL", "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "capability": list(cap), "arch_ok": arch_ok, "driver_ge_torch_cuda": drv_ok,
            "detail": f"torch {torch.__version__} / 빌드 CUDA {torch.version.cuda} / {sm} 커널 {'있음' if arch_ok else '없음'}"
                      f" / 드라이버 CUDA {'≥' if drv_ok else '<'} 빌드 CUDA"}
    # --- /제공 코드 ---


def check_cudnn_and_path() -> dict:
    """cuDNN 로드를 별도 프로세스로 시험. venv 경로는 /workspace 아래 영문 경로 권장."""
    # --- 제공 코드(4, 수정 불필요): sys.executable로 'import torch; print(torch.backends.cudnn.version())' 실행, 실패 시 FAIL ---
    prefix = sys.prefix
    ascii_ok = prefix.isascii()
    try:
        p = subprocess.run([sys.executable, "-c", "import torch;print(torch.backends.cudnn.version())"],
                           capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace")
        ver = p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else ""
    except subprocess.TimeoutExpired:
        ver = ""
    ok = ver.isdigit()
    return {"status": "OK" if ok else "FAIL", "cudnn": ver, "venv_ascii": ascii_ok,
            "detail": (f"cuDNN {ver}" if ok else "cuDNN 로드 실패")
                      + f" / venv 경로 {'ASCII' if ascii_ok else '비ASCII 포함'}"
                      + ("" if ok else " — 조치: /workspace 아래 영문 경로로 이동")}
    # --- /제공 코드 ---


def bench(seconds: float = 3.0) -> dict:
    """bf16 matmul TFLOPS와 device-to-device 복사 대역폭(GB/s)."""
    # --- 제공 코드(3, 수정 불필요): 4096×4096 bf16 matmul 반복 → TFLOPS, 1GiB 텐서 copy_ 반복 → GB/s (synchronize 필수) ---
    import torch
    dev = "cuda"
    n = 4096
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    it, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a @ b
        it += 1
    torch.cuda.synchronize()
    tflops = 2 * n**3 * it / (time.perf_counter() - t0) / 1e12
    x = torch.empty(256 * 1024**2, device=dev, dtype=torch.float32)  # 1 GiB
    y = torch.empty_like(x)
    y.copy_(x)
    torch.cuda.synchronize()
    it, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        y.copy_(x)
        it += 1
    torch.cuda.synchronize()
    gbps = 2 * x.numel() * 4 * it / (time.perf_counter() - t0) / 1e9  # 읽기+쓰기
    peak = torch.cuda.max_memory_allocated() / 1024**3
    del a, b, x, y
    torch.cuda.empty_cache()
    return {"status": "OK", "bf16_tflops": round(tflops, 1), "copy_gbps": round(gbps, 0),
            "peak_alloc_gib": round(peak, 2),
            "detail": f"bf16 matmul {tflops:.1f} TFLOPS / 복사 대역폭 {gbps:.0f} GB/s"}
    # --- /제공 코드 ---


def check_disk(path: Path) -> dict:
    free = shutil.disk_usage(path).free / 1024**3
    return {"status": "OK" if free >= MIN_FREE_GB else "FAIL", "free_gb": round(free, 1),
            "detail": f"{path} 여유 {free:.0f} GB (기준 {MIN_FREE_GB} GB)"}


def compare_reference(smi: dict, b: dict) -> list[dict]:
    rows = []
    for name, (vram, tf, bw) in REFERENCE.items():
        rows.append({"reference": name, "ref_vram_mib": vram, "ref_tflops": tf, "ref_bw": bw,
                     "vram_ratio": round(smi.get("memory.total", 0) / vram, 2),
                     "tflops_ratio": round(b.get("bf16_tflops", 0) / tf, 2),
                     "bw_ratio": round(b.get("copy_gbps", 0) / bw, 2)})
    return rows


def self_check(res: dict) -> None:
    for k in ("nvidia_smi", "nvcc", "torch", "cudnn", "bench", "disk"):
        assert res[k]["status"] in {"OK", "WARN", "FAIL"}, k
    assert res["torch"]["status"] == "OK", "torch CUDA/아키텍처 불일치"
    assert res["cudnn"]["status"] == "OK", "cuDNN 로드 실패(경로 확인)"
    assert res["bench"]["bf16_tflops"] > 1, "matmul 실측값 비정상"
    print("[PASS] ex00b self-check 통과")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="/workspace" if Path("/workspace").exists() else str(HERE))
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    try:
        res = {"nvidia_smi": query_nvidia_smi(), "nvcc": check_nvcc()}
        res["torch"] = check_torch(res["nvidia_smi"])
        res["cudnn"] = check_cudnn_and_path()
        res["bench"] = bench(a.seconds) if res["torch"]["status"] == "OK" else {"status": "FAIL", "detail": "생략"}
        res["disk"] = check_disk(Path(a.workspace))
    except NotImplementedError as e:
        print(f"[안내] 아직 구현되지 않은 TODO: {e}")
        sys.exit(1)
    res["reference"] = compare_reference(res["nvidia_smi"], res["bench"])
    for k in ("nvidia_smi", "nvcc", "torch", "cudnn", "bench", "disk"):
        print(f"  [{res[k]['status']:4s}] {k:10s} {res[k]['detail']}")
    for r in res["reference"]:
        print(f"  vs {r['reference']}: VRAM ×{r['vram_ratio']} / TFLOPS ×{r['tflops_ratio']} / 대역폭 ×{r['bw_ratio']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    self_check(res)


if __name__ == "__main__":
    main()
