"""Models served by vLLM-Omni (`vllm serve <repo> --omni`) or SGLang-Omni (`sgl-omni serve`):
the runner starts the server from its own env, waits for /health, then posts one
/v1/audio/speech request per segment (WAV back).
The server's GPU memory is read from nvidia-smi (torch in the runner process sees none)."""

import atexit
import io
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = 8097


def start(repo: str, extra: tuple[str, ...] = (), server: str = "vllm") -> dict:
    bindir = Path(sys.executable).parent
    if server == "vllm":
        cmd = [str(bindir / "vllm"), "serve", repo, "--omni", "--trust-remote-code"]
    else:  # sglang-omni
        cmd = [str(bindir / "sgl-omni"), "serve", "--model-path", repo]
    # FlashInfer (vLLM) and DeepGEMM (SGLang) JIT-compile kernels and look for a CUDA toolkit;
    # the box has none, but NVIDIA's cu13 wheels in this env are one (bin/nvcc, include/, nvvm/).
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"  # ninja and friends from this env
    toolkit = _toolkit()
    if toolkit is not None:
        env["CUDA_HOME"] = env["CUDA_PATH"] = str(toolkit)
        env["PATH"] = f"{toolkit / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    proc = subprocess.Popen(
        [*cmd, "--port", str(PORT), *extra],
        stdout=sys.stderr,
        stderr=sys.stderr,
        start_new_session=True,
        env=env,
    )

    def stop():
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)

    atexit.register(stop)
    url = f"http://127.0.0.1:{PORT}"
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited {proc.returncode} while loading")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
                if resp.status == 200:
                    break
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    return {"url": url, "repo": repo, "proc": proc, "peak_gb": gpu_used_gb}


def _toolkit() -> Path | None:
    """A CUDA_HOME built from the env's NVIDIA cu13 wheels: they hold bin/nvcc, include/ and
    nvvm/ but only lib/libX.so.N, while JIT builds link `-L$CUDA_HOME/lib64 -lcudart`. So
    <env>/cuda gets the same folders as links, plus lib64 with the unversioned names."""
    wheels = next(Path(sys.prefix).glob("lib/python3*/site-packages/nvidia/cu13"), None)
    if wheels is None or not (wheels / "bin" / "nvcc").exists():
        return None
    home = Path(sys.prefix) / "cuda"
    (home / "lib64").mkdir(parents=True, exist_ok=True)
    for name in ("bin", "include", "nvvm"):
        link = home / name
        if not link.exists():
            link.symlink_to(wheels / name)
    for lib in (wheels / "lib").iterdir():
        for name in {lib.name, lib.name.split(".so")[0] + ".so"}:
            link = home / "lib64" / name
            if not link.exists():
                link.symlink_to(lib)
    return home


def speech(engine: dict, payload: dict, timeout: float = 600) -> tuple:
    import soundfile as sf

    body = json.dumps({"model": engine["repo"], "response_format": "wav", **payload}).encode()
    request = urllib.request.Request(
        f"{engine['url']}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:400]}") from exc
    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1), rate


def data_url(path: str) -> str:
    import base64

    return "data:audio/wav;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def gpu_used_gb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return round(int(out.split()[0]) / 1024, 2)
    except Exception:  # noqa: BLE001 - just not reported
        return None
