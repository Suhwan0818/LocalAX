"""
ex00a_local_env_check_q.py — 로컬 개발 환경 자가 점검

학습 목표
  - uv·Python·Git 신원·.gitignore·.env 상태를 한 번에 점검
  - 키 값은 마스킹해서만 출력, .env가 Git에 추적되는지 판정
실행(수정 없이 실행만 하는 사전 점검): uv run python ex00a_local_env_check_q.py [--out results/ex00a_local_env_check.json]
대응 슬라이드: M1 개발 환경 '재현 가능한 작업 폴더'(35) · 'Git 최소 명령 6개'(36) · '키는 .env에만 보관'(39)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 콘솔 인코딩 통일(Linux는 기본 UTF-8)

from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
PY_RANGE = ((3, 11), (3, 12))           # 허용 Python 범위(포함) · RunPod 이미지 torch는 3.12에 설치됨
OPTIONAL_KEYS = ["HF_TOKEN", "GEMINI_API_KEY", "TAVILY_API_KEY"]
GITIGNORE_MUST = [".env", ".venv", "__pycache__"]


def run(cmd: list[str]) -> tuple[int, str]:
    """명령 실행 → (반환코드, stdout+stderr). 명령이 없으면 (127, '')."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                           encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout + p.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 127, ""


def mask(v: str) -> str:
    """앞 4글자만 노출. 짧으면 전부 가림."""
    return "****" if len(v) <= 6 else v[:4] + "*" * 4 + f"({len(v)}자)"


def check_python() -> dict:
    # --- 제공 코드(1, 수정 불필요): sys.version_info가 PY_RANGE 안이면 OK, 아니면 FAIL + 이미지 Python으로 venv 재생성 조치 ---
    v = sys.version_info[:2]
    ok = PY_RANGE[0] <= v <= PY_RANGE[1]
    return {"status": "OK" if ok else "FAIL",
            "detail": f"Python {v[0]}.{v[1]} ({sys.executable})"
                      + ("" if ok else " — 조치: uv venv --clear --system-site-packages --python /usr/bin/python3.12 후 uv sync --inexact")}
    # --- /제공 코드 ---


def check_uv() -> dict:
    code, out = run(["uv", "--version"])
    venv_ok = ".venv" in sys.prefix or os.environ.get("VIRTUAL_ENV", "") != ""
    if code != 0:
        return {"status": "FAIL", "detail": "uv 미설치 — 조치: 설치 스크립트 실행 후 터미널 재시작"}
    lock = (HERE / "uv.lock").exists()
    return {"status": "OK" if (lock and venv_ok) else "WARN",
            "detail": f"{out} / uv.lock {'있음' if lock else '없음'} / venv {'사용' if venv_ok else '미사용(uv run 권장)'}"}


def check_git() -> dict:
    # --- 제공 코드(2, 수정 불필요): git --version, user.name/user.email 설정 여부 확인. 둘 중 하나라도 비면 FAIL ---
    code, ver = run(["git", "--version"])
    if code != 0:
        return {"status": "FAIL", "detail": "git 미설치 — 조치: apt-get update && apt-get install -y git"}
    _, name = run(["git", "config", "--global", "user.name"])
    _, email = run(["git", "config", "--global", "user.email"])
    ok = bool(name) and bool(email)
    return {"status": "OK" if ok else "FAIL",
            "detail": f"{ver} / user.name={'설정' if name else '없음'} / user.email={'설정' if email else '없음'}"
                      + ("" if ok else " — 조치: git config --global user.name/user.email")}
    # --- /제공 코드 ---


def check_gitignore(root: Path) -> dict:
    # --- 제공 코드(3, 수정 불필요): root/.gitignore에 GITIGNORE_MUST 패턴이 모두 있는지 확인 ---
    gi = root / ".gitignore"
    if not gi.exists():
        return {"status": "FAIL", "detail": ".gitignore 없음 — 조치: .env, .venv, __pycache__ 추가"}
    lines = {ln.strip().rstrip("/") for ln in gi.read_text(encoding="utf-8").splitlines()}
    missing = [p for p in GITIGNORE_MUST if p not in lines]
    return {"status": "OK" if not missing else "FAIL",
            "detail": "필수 패턴 모두 포함" if not missing else f"누락: {missing}"}
    # --- /제공 코드 ---


def check_env_tracked(root: Path) -> dict:
    """Git 저장소이면 .env가 추적(tracked)되는지 확인. 추적되면 치명(FAIL)."""
    code, _ = run(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"])
    if code != 0:
        return {"status": "WARN", "detail": "Git 저장소 아님 — 조치: 본인 GitHub 저장소(w01)를 clone한 폴더에서 다시 실행"}
    _, out = run(["git", "-C", str(root), "ls-files", ".env"])
    if out:
        return {"status": "FAIL", "detail": ".env가 Git에 추적됨 — 조치: 키 revoke 후 git rm --cached .env"}
    return {"status": "OK", "detail": ".env 미추적"}


def check_dotenv(root: Path) -> dict:
    # --- 제공 코드(4, 수정 불필요): .env 존재 확인, OPTIONAL_KEYS 값은 mask()로만 출력. 없으면 WARN ---
    envp = root / ".env"
    if not envp.exists():
        ex = (root / ".env.example").exists()
        return {"status": "WARN",
                "detail": ".env 없음 — 조치: cp .env.example .env" if ex else ".env·.env.example 모두 없음"}
    vals = dotenv_values(envp)
    parts = [f"{k}={mask(vals[k])}" if vals.get(k) else f"{k}=(빈 값)" for k in OPTIONAL_KEYS]
    status = "OK" if vals.get("HF_TOKEN") else "WARN"
    return {"status": status, "detail": " / ".join(parts)}
    # --- /제공 코드 ---


def check_disk(root: Path, min_gb: float = 20) -> dict:
    free = shutil.disk_usage(root).free / 1024**3
    return {"status": "OK" if free >= min_gb else "FAIL",
            "detail": f"{free:.0f} GB 여유 (기준 {min_gb:.0f} GB: 모델·캐시)"}


def check_path_ascii(root: Path) -> dict:
    """프로젝트·venv 경로 ASCII 점검. /workspace 아래 영문 경로 권장."""
    bad = [c for c in str(root.resolve()) + sys.prefix if not c.isascii()]
    return {"status": "OK" if not bad else "WARN",
            "detail": "경로 ASCII" if not bad else
            f"비ASCII 문자 {len(bad)}개 포함 — GPU 실습 전 영문 경로(예: /workspace/w01)로 옮길 것"}


def run_all(root: Path) -> dict:
    return {
        "path_ascii": check_path_ascii(root),
        "python": check_python(),
        "uv": check_uv(),
        "git": check_git(),
        "gitignore": check_gitignore(root),
        "env_tracked": check_env_tracked(root),
        "dotenv": check_dotenv(root),
        "disk": check_disk(root),
    }


def self_check(res: dict) -> None:
    assert set(res) == {"path_ascii", "python", "uv", "git", "gitignore", "env_tracked", "dotenv", "disk"}
    for k, r in res.items():
        assert r["status"] in {"OK", "WARN", "FAIL"}, k
        assert isinstance(r["detail"], str)
        # 키 원문이 출력에 섞이면 안 됨
    vals = dotenv_values(HERE / ".env") if (HERE / ".env").exists() else {}
    blob = json.dumps(res, ensure_ascii=False)
    for k, v in vals.items():
        if v and len(v) > 6:
            assert v not in blob, f"{k} 원문 노출"
    fails = [k for k, r in res.items() if r["status"] == "FAIL"]
    if fails:
        # 코드 오류가 아니라 환경 문제 → Traceback 대신 조치 안내만 출력
        print(f"\n[환경 조치 필요] FAIL 항목: {', '.join(fails)}")
        print("  - 코드는 수정하지 말 것. 위 표의 '조치' 문구대로 환경을 고친 뒤 다시 실행")
        sys.exit(1)
    print("[PASS] ex00a self-check 통과")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    root = Path(a.root)
    try:
        res = run_all(root)
    except NotImplementedError as e:
        print(f"[안내] 아직 구현되지 않은 TODO: {e}")
        sys.exit(1)
    print(f"ex00a 로컬 환경 점검 — {platform.system()} {platform.release()}")
    for k, r in res.items():
        print(f"  [{r['status']:4s}] {k:12s} {r['detail']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    self_check(res)


if __name__ == "__main__":
    main()
