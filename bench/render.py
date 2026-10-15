"""Step 3: render each model's engine script to audio, one model at a time.

Per model: create `envs/<model>/` with uv on first use (its own Python and pins, CUDA 12.8 torch
for RTX 50-series cards), write `work.jsonl` (the engine script minus empty segments), start
`runners/run_<model>.py` in that env and save each segment as it comes (`segments/<id>.wav`), so
a stopped run resumes. A model that stalls is killed; failures are recorded in run.json and the
run moves on. Model downloads go to $HF_HOME (torch and other caches inside it).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .assemble import assemble
from .registry import ALL_MODELS

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parent
ENVS = ROOT / "envs"
# Every model download: $HF_HOME, else Hugging Face's default cache.
DOWNLOADS = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")

STALL_S = 600  # no event for this long after loading: the model is stuck
LOAD_S = 3600  # first load includes downloading the weights
MODEL_S = 3 * 3600


# Windows Smart App Control blocks native wheels it hasn't seen widely yet ("An Application
# Control policy has blocked this file"): seen with tokenizers 0.23.3 and pydantic_core. So
# every env resolves only packages published before EXCLUDE_NEWER (Spec.exclude_newer to move
# it for a model released later), and tokenizers stays at 0.22.2 with transformers 5.2.0, the
# newest that accepts it (verified in envs/chatterbox_turbo). Specs that pin their own
# transformers keep that pin.
WINDOWS = os.name == "nt"
LINUX_DATE = "2026-10-14"
# Smart App Control only exists on Windows: elsewhere resolve freely up to the run date.
EXCLUDE_NEWER = "2026-08-01" if WINDOWS else LINUX_DATE
# numpy < 2.5: no numba released before EXCLUDE_NEWER supports 2.5, so uv fell back to a 2021
# numba that doesn't build; and torch 2.9.1 broke on import under numpy 2.5.4.
SAC_SAFE = ("tokenizers==0.22.2", "numpy<2.5") if WINDOWS else ()
TORCH_28 = ("torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0")
# Linux: torch only; each model's own transformers pin stands (5.2.0 broke OmniVoice, StyleTTS2).
DEFAULT_OVERRIDES = (*TORCH_28, *SAC_SAFE, "transformers==5.2.0") if WINDOWS else TORCH_28


@dataclass(frozen=True)
class Spec:
    python: str
    packages: tuple[str, ...]
    # torch pins that work on the RTX 5090 (sm_120 needs CUDA 12.8 wheels, torch >= 2.7)
    overrides: tuple[str, ...] = DEFAULT_OVERRIDES
    device: str = "cuda"
    note: str = ""
    # A git repo with no pip package: shallow-cloned to envs/<model>/src, its requirements.txt
    # installed, and the runner puts src/ on sys.path.
    repo: str = ""
    exclude_newer: str = EXCLUDE_NEWER
    # How the repo goes in: "requirements" (its requirements.txt, if any), "package" (the
    # checkout itself, `pip install src`), "sync" (`uv sync --frozen` from the repo's own
    # uv.lock into envs/<model>; packages/overrides unused), or "none" (Spec.packages only).
    repo_install: str = "requirements"
    submodules: bool = False
    # "auto" lets uv match the driver (vLLM's own CUDA build); "index" = PyTorch's cu128 index as
    # an extra index instead, for packages (torchtune) the torch-backend mode looks for there only
    torch_backend: str = "cu128"
    repo_extras: str = ""  # with repo_install "package": install src[extras]
    prerelease: bool = False


# Order = run order: small/fast first, so a stopped night still has the most results.
CHATTERBOX = ("chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox",)
# Chatterbox keeps the set it first rendered with (no date cutoff, no numpy cap): under the
# cutoff its s3tokenizer got float64 audio against float32 mel filters and every segment failed.
CHATTERBOX_SPEC = dict(
    overrides=(*TORCH_28, "tokenizers==0.22.2", "transformers==5.2.0"), exclude_newer="2026-10-14"
)
# Breeze's pinned stack (its README); qwen-tts runs on the same one.
PROJECT_TORCH = ("torch==2.9.1", "torchaudio==2.9.1", *SAC_SAFE, "transformers==4.57.3")
KOKORO = (
    "kokoro>=0.9.4",
    "misaki>=0.9.4",
    "num2words>=0.5.14",
    "spacy>=3.8",
    "phonemizer>=3.4",
    "espeakng-loader>=0.2.4",
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
)
VLLM_OMNI = ("vllm==0.30.0", "vllm-omni==0.30.0")
# The JIT kernel builds (FlashInfer, SGLang/DeepGEMM) use NVIDIA's pip toolkit: its compiler
# must match torch's CUDA 13.2 runtime headers (uv otherwise takes nvcc 13.4).
CUDA_132 = (
    "nvidia-cuda-nvcc==13.2.*",
    "nvidia-cuda-crt==13.2.*",
    "nvidia-cuda-nvdisasm==13.2.*",
    "nvidia-nvvm==13.2.*",
    "nvidia-nvjitlink==13.2.*",
)  # nvvm 13.4 emitted PTX 9.4
# CosyVoice's requirements.txt minus what inference never imports (deepspeed, tensorrt, gradio,
# grpc, fastapi, tensorboard) and its torch 2.3 pin (no sm_120); diffusers unpinned (0.29 needs
# a huggingface_hub too old for transformers 4.51).
COSYVOICE = (
    "torch",
    "torchaudio",
    "rich",
    "gdown",
    "wget",
    "setuptools<81",
    "conformer==0.3.2",
    "diffusers",
    "hydra-core==1.3.2",
    "HyperPyYAML==1.2.3",
    "inflect",
    "librosa",
    "lightning",
    "modelscope",
    "omegaconf==2.3.0",
    "onnx",
    "onnxruntime-gpu",
    "openai-whisper",
    "pyarrow",
    "pyworld",
    "transformers==4.51.3",
    "x-transformers==2.11.24",
    "wetext==0.0.4",
    "einops",
    "tiktoken",
    "regex",
    "scipy",
    "matplotlib",
    "huggingface_hub",
)
SPECS: dict[str, Spec] = {
    "kokoro": Spec("3.12", KOKORO),
    "chatterbox_nano": Spec("3.11", CHATTERBOX, **CHATTERBOX_SPEC),
    "chatterbox_turbo": Spec("3.11", CHATTERBOX, **CHATTERBOX_SPEC),
    "chatterbox": Spec("3.11", CHATTERBOX, **CHATTERBOX_SPEC),
    "f5tts": Spec("3.11", ("f5-tts",)),
    "qwen3_tts": Spec("3.12", ("qwen-tts==0.1.1", "transformers==4.57.3"), PROJECT_TORCH),
    # nari-tts pins triton 3.2.0 on Linux, which torch 2.8's triton 3.4.0 contradicts
    "dia": Spec(
        "3.11",
        ("nari-tts @ git+https://github.com/nari-labs/dia",),
        (*DEFAULT_OVERRIDES, "triton==3.4.0"),
    ),
    "voxcpm2": Spec("3.11", ("voxcpm",)),
    "omnivoice": Spec("3.11", ("omnivoice",)),
    "kitten_tts2": Spec("3.11", ("kittenml",)),
    # neucodec needs torchtune, which isn't on PyTorch's index: "index" mode
    # torchao 0.14+ needs torch >= 2.9 (ScalingType): 0.13 is torch 2.8's
    "neutts_nano": Spec(
        "3.11",
        ("neutts", "espeakng-loader", "phonemizer"),
        (*TORCH_28, "torchao==0.13.0", "datasets>=3"),
        torch_backend="index",
    ),
    "neutts_air": Spec(
        "3.11",
        ("neutts", "espeakng-loader", "phonemizer"),
        (*TORCH_28, "torchao==0.13.0", "datasets>=3"),
        torch_backend="index",
    ),
    "styletts2": Spec("3.11", ("styletts2", "espeakng-loader", "phonemizer")),
    "indextts2": Spec(
        "3.11", ("indextts @ git+https://github.com/index-tts/index-tts", "huggingface_hub")
    ),
    "maya1": Spec("3.11", ("transformers", "accelerate", "snac")),
    "moss_tts": Spec(
        "3.12",
        ("transformers==5.0.0", "accelerate", "torchcodec", "torchaudio", "nvidia-npp-cu12"),
        # torchcodec 0.11 is built against a newer torch (undefined torch_dtype_float4_e2m1fn_x2)
        (
            "torch==2.9.1",
            "torchaudio==2.9.1",
            *SAC_SAFE,
            "transformers==5.0.0",
            "torchcodec>=0.8,<0.10",
        ),
        note="8B, bf16",
    ),
    # Native Linux (2026-10-14): no Smart App Control here, so no date cutoff beyond today and
    # no tokenizers/numpy caps; torch stays 2.8 (cu128, sm_120) unless the model pins its own.
    # from a checkout: its pyproject packages only "zonos", not zonos.backbone & co.
    "zonos": Spec(
        "3.12",
        (
            "torch",
            "torchaudio",
            "inflect",
            "kanjize",
            "phonemizer",
            "sudachidict-full",
            "sudachipy",
            "transformers",
            "huggingface-hub",
            "espeakng-loader",
            "setuptools",
            "packaging",
        ),
        TORCH_28,
        repo="https://github.com/Zyphra/Zonos",
        repo_install="none",
        exclude_newer=LINUX_DATE,
    ),
    "moss_tts_nano": Spec(
        "3.12",
        ("torch", "torchaudio", "transformers==4.57.1", "sentencepiece", "einops"),
        (*TORCH_28, "transformers==4.57.1"),
        exclude_newer=LINUX_DATE,
    ),
    "pocket_tts": Spec(
        "3.12",
        ("pocket-tts",),
        TORCH_28,
        device="cpu",
        exclude_newer=LINUX_DATE,
        note="CPU, as upstream intends",
    ),
    "csm_1b": Spec(
        "3.12",
        ("transformers==4.57.3", "accelerate"),
        (*TORCH_28, "transformers==4.57.3"),
        exclude_newer=LINUX_DATE,
    ),
    "orpheus": Spec(
        "3.12",
        ("transformers==4.57.3", "accelerate", "snac"),
        (*TORCH_28, "transformers==4.57.3"),
        exclude_newer=LINUX_DATE,
    ),
    "cosyvoice3": Spec(
        "3.10",
        COSYVOICE,
        (*TORCH_28, "ruamel.yaml<0.19", "numpy<2", "setuptools<81"),  # pkg_resources
        repo="https://github.com/FunAudioLLM/CosyVoice",
        repo_install="none",
        submodules=True,
        exclude_newer=LINUX_DATE,
    ),
    "fish_s2": Spec(
        "3.12",
        ("huggingface_hub",),
        # pyaudio (api_client.py only) needs PortAudio headers: never installed
        (*TORCH_28, "pyaudio; sys_platform == 'never'"),
        repo="https://github.com/fishaudio/fish-speech",
        repo_install="package",
        exclude_newer=LINUX_DATE,
    ),
    "higgs_v3": Spec(
        "3.12",
        VLLM_OMNI,
        CUDA_132,
        torch_backend="auto",
        exclude_newer=LINUX_DATE,
        note="vLLM-Omni server; peak memory from nvidia-smi",
    ),
    "step_audio_editx": Spec(
        "3.12",
        (),
        (),
        repo="https://github.com/stepfun-ai/Step-Audio-EditX",
        repo_install="sync",
        exclude_newer=LINUX_DATE,
        note="upstream uv.lock (torch 2.9.1, vLLM 0.14 rc); tags edited in after a plain clone",
    ),
    "voxtral_tts": Spec(
        "3.12",
        VLLM_OMNI,
        (),
        torch_backend="auto",
        exclude_newer=LINUX_DATE,
        note="vLLM-Omni server; presets only (open weights have no encoder)",
    ),
    "higgs_v2": Spec(
        "3.12",
        ("torch", "torchaudio", "librosa", "transformers>=5.3", "accelerate"),
        TORCH_28,
        exclude_newer=LINUX_DATE,
    ),
    # xcodec2 needs torchtune (not on PyTorch's index: "index" mode) and pins its own stack
    # transformers 5 refuses XCodec2's nested from_pretrained ("meta device context")
    "llasa_8b": Spec(
        "3.11",
        ("xcodec2", "accelerate"),
        (*TORCH_28, "torchao==0.13.0", "transformers==4.57.3"),
        torch_backend="index",
        exclude_newer=LINUX_DATE,
    ),
    "miso_tts": Spec(
        "3.11",
        (),
        (*TORCH_28, "torchao==0.13.0"),
        repo="https://github.com/MisoLabsAI/MisoTTS",
        exclude_newer=LINUX_DATE,
        torch_backend="index",
        note="tokenizer from the ungated unsloth/Llama-3.2-1B mirror",
    ),
    "orpheus_clone": Spec(
        "3.12",
        ("transformers==4.57.3", "accelerate", "snac"),
        (*TORCH_28, "transformers==4.57.3"),
        exclude_newer=LINUX_DATE,
    ),
    "t5gemma_tts": Spec(
        "3.11",
        (),
        (*TORCH_28, "torchao==0.13.0"),
        repo="https://github.com/Aratako/T5Gemma-TTS",
        torch_backend="index",
        exclude_newer=LINUX_DATE,
    ),
    # protobuf: sglang-omni's own tool.uv override (descript-audiotools pins <3.20), which
    # `uv pip install` of the package doesn't apply
    "drama3": Spec(
        "3.12",
        (),
        ("protobuf>=6.31.1,<7", *CUDA_132),
        repo="https://github.com/sgl-project/sglang-omni",
        repo_install="package",
        repo_extras="fishaudio",
        prerelease=True,
        torch_backend="auto",
        exclude_newer=LINUX_DATE,
        note="SGLang-Omni server, int4 checkpoint; peak memory from nvidia-smi",
    ),
    "breeze_tts2": Spec(
        "3.12",
        ("qwen-tts==0.1.1", "transformers==4.57.3", "huggingface_hub"),
        PROJECT_TORCH,
        note="eager runtime (BREEZE_TTS_FAST off), the vendor/breeze-tts submodule's code",
    ),
}


def _env_python(model: str) -> Path:
    return (
        ENVS / model / "Scripts" / "python.exe"
        if os.name == "nt"
        else ENVS / model / "bin" / "python"
    )


def _child_env(spec: Spec) -> dict[str, str]:
    env = dict(os.environ)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    env.update(
        HF_HOME=str(DOWNLOADS),
        TORCH_HOME=str(DOWNLOADS / "torch"),
        XDG_CACHE_HOME=str(DOWNLOADS / "cache"),
        MODELSCOPE_CACHE=str(DOWNLOADS / "modelscope"),
        VT_DEVICE=spec.device,
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
    )
    env.pop("VIRTUAL_ENV", None)
    # Gated repos: HF_TOKEN (~/.bashrc), else an `hf auth login` token file.
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if "HF_TOKEN" not in env and token_file.exists():
        env["HF_TOKEN"] = token_file.read_text(encoding="utf-8").strip()
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return env


def ensure_env(model: str, spec: Spec, log) -> None:
    stamp = ENVS / model / ".ready"
    parts = [spec.python, spec.packages, spec.overrides, spec.repo, spec.exclude_newer]
    if spec.repo_install != "requirements" or spec.submodules or spec.torch_backend != "cu128":
        parts += [spec.repo_install, spec.submodules, spec.torch_backend]  # older stamps hold
    if spec.repo_extras or spec.prerelease:
        parts += [spec.repo_extras, spec.prerelease]
    key = json.dumps(parts)
    if stamp.exists() and stamp.read_text(encoding="utf-8") == key:
        return
    log(f"creating envs/{model} (python {spec.python})")
    subprocess.run(
        ["uv", "venv", str(ENVS / model), "--python", spec.python, "--clear"],
        check=True,
        capture_output=True,
        text=True,
    )
    overrides = ENVS / model / "overrides.txt"
    overrides.write_text("\n".join(spec.overrides) + "\n", encoding="utf-8")
    extra: list[str] = []
    if spec.repo:
        src = ENVS / model / "src"
        if src.exists():
            shutil.rmtree(src)
        clone = ["git", "clone", "-q", "--depth", "1"]
        if spec.submodules:
            clone += ["--recurse-submodules", "--shallow-submodules"]
        subprocess.run([*clone, spec.repo, str(src)], check=True)
        if spec.repo_install == "sync":
            result = subprocess.run(
                ["uv", "sync", "--frozen", "--project", str(src), "--python", spec.python],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(ENVS / model)},
            )
            log(result.stdout[-3000:] + result.stderr[-6000:])
            if result.returncode != 0:
                raise RuntimeError(
                    "uv sync failed: " + (result.stderr.strip().splitlines() or ["?"])[-1]
                )
            stamp.write_text(key, encoding="utf-8")
            return
        if spec.repo_install == "package":
            extra = [f"{src}[{spec.repo_extras}]" if spec.repo_extras else str(src)]
        elif spec.repo_install == "requirements" and (src / "requirements.txt").exists():
            extra = ["-r", str(src / "requirements.txt")]
    cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        str(_env_python(model)),
        *(
            [
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu128",
                "--index-strategy",
                "unsafe-best-match",
            ]
            if spec.torch_backend == "index"
            else ["--torch-backend", spec.torch_backend]
        ),
        "--override",
        str(overrides),
        "--exclude-newer",
        spec.exclude_newer,
        *(["--prerelease=allow"] if spec.prerelease else []),
        "numpy",
        "soundfile",
        *spec.packages,
        *extra,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log(result.stdout[-3000:] + result.stderr[-6000:])
    if result.returncode != 0:
        raise RuntimeError("install failed: " + (result.stderr.strip().splitlines() or ["?"])[-1])
    stamp.write_text(key, encoding="utf-8")


def _is_empty(row: dict, prefix: str) -> bool:
    text = row["text"][len(prefix) :] if prefix and row["text"].startswith(prefix) else row["text"]
    return not text.strip()


def run_runner(model: str, work: Path, out_dir: Path, log) -> dict:
    """Run runners/run_<model>.py in its env over work.jsonl, saving <id>.wav into out_dir.
    Returns {"load_s", "stats": {id: {compute_s, audio_s, peak_gb}}, "errors": {id: msg},
    "notes": [...]}."""
    spec = SPECS[model]
    ensure_env(model, spec, log)
    cmd = [
        str(_env_python(model)),
        str(TOOL / "runners" / f"run_{model}.py"),
        str(work),
        str(out_dir),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_child_env(spec),
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(TOOL / "runners"),
    )
    stderr_tail: list[str] = []

    def drain():
        for line in proc.stderr:
            stderr_tail.append(line)
            del stderr_tail[:-200]
            log("  | " + line.rstrip())

    events: list[dict] = []
    last = [time.monotonic()]

    def read():
        for line in proc.stdout:
            last[0] = time.monotonic()
            if line.startswith('{"event"'):
                events.append(json.loads(line))
            log("  " + line.rstrip())

    threading.Thread(target=drain, daemon=True).start()
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    started, why_killed = time.monotonic(), None
    while proc.poll() is None:
        time.sleep(2)
        loaded = any(e["event"] == "loaded" for e in events)
        idle = time.monotonic() - last[0]
        if (loaded and idle > STALL_S) or (not loaded and idle > LOAD_S):
            why_killed = f"no progress for {idle:.0f}s"
        elif time.monotonic() - started > MODEL_S:
            why_killed = f"model time limit ({MODEL_S}s)"
        if why_killed:
            proc.kill()
            break
    proc.wait()
    reader.join(timeout=5)
    out = {"load_s": None, "stats": {}, "errors": {}, "notes": []}
    for e in events:
        if e["event"] == "loaded":
            out["load_s"] = e["load_s"]
        elif e["event"] == "segment":
            out["stats"][e["id"]] = {k: e[k] for k in ("compute_s", "audio_s", "peak_gb")}
        elif e["event"] == "error":
            out["errors"][e["id"]] = e["error"]
    if why_killed:
        out["notes"].append(f"stopped: {why_killed}")
    if proc.returncode not in (0, None) and not any(e["event"] == "done" for e in events):
        tail = "".join(stderr_tail[-15:]).strip()
        out["notes"].append(f"runner exited {proc.returncode}: {tail[-1500:]}")
    return out


def _row_hash(row: dict) -> str:
    """What a segment was rendered from: a changed engine script re-renders it."""
    key = {k: v for k, v in row.items() if k not in ("id", "empty")}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def render_model(
    model: str, tests: list[str], output: Path, listen_dir: Path, log, render: bool = True
) -> dict[str, dict]:
    """Render (resuming) every test's engine script for one model in a single runner process,
    then assemble each test. Returns {test: what goes in that test's run.json}."""
    card, spec = ALL_MODELS[model], SPECS[model]
    per_test: dict[str, tuple[list[dict], dict]] = {}
    work: list[dict] = []
    for test in tests:
        model_dir = output / test / model
        rows = [
            json.loads(x)
            for x in (model_dir / f"{test}.engine.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        for row in rows:
            row["empty"] = _is_empty(row, card.speaker_prefix)
        stats_path = model_dir / "segment-stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        for row in rows:  # rendered from something else (or the segment is now empty): redo
            wav = model_dir / "segments" / f"{row['id']}.wav"
            if wav.exists() and (
                row["empty"] or stats.get(row["id"], {}).get("input") != _row_hash(row)
            ):
                wav.unlink()
                stats.pop(row["id"], None)
        per_test[test] = (rows, stats)
        # one work list for every test: ids are paths under output/
        work += [{**r, "id": f"{test}/{model}/segments/{r['id']}"} for r in rows if not r["empty"]]
    todo = [r for r in work if not (output / f"{r['id']}.wav").exists()]
    run = {"load_s": None, "stats": {}, "errors": {}, "notes": []}
    failed = None
    if todo and render:
        work_path = output / ".work" / f"{model}.jsonl"
        work_path.parent.mkdir(parents=True, exist_ok=True)
        work_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in work), encoding="utf-8"
        )
        log(f"rendering {len(todo)} segments")
        try:
            run = run_runner(model, work_path, output, log)
        except Exception as exc:  # noqa: BLE001 - recorded in run.json, next model
            failed = f"{type(exc).__name__}: {exc}"[:800]

    infos = {}
    for test, (rows, stats) in per_test.items():
        model_dir = output / test / model
        prefix = f"{test}/{model}/segments/"
        new = {k[len(prefix) :]: v for k, v in run["stats"].items() if k.startswith(prefix)}
        hashes = {r["id"]: _row_hash(r) for r in rows}
        for sid, entry in new.items():
            entry["input"] = hashes[sid]
        if new:
            stats.update(new)
            (model_dir / "segment-stats.json").write_text(
                json.dumps(stats, indent=1), encoding="utf-8"
            )
        errors = {k[len(prefix) :]: v for k, v in run["errors"].items() if k.startswith(prefix)}
        seg_dir = model_dir / "segments"
        assembled = assemble(seg_dir, rows, listen_dir / model, test)
        done = [r["id"] for r in rows if not r["empty"] and (seg_dir / f"{r['id']}.wav").exists()]
        compute = sum(stats[i]["compute_s"] for i in done if i in stats)
        audio = sum(stats[i]["audio_s"] for i in done if i in stats)
        peaks = [
            stats[i]["peak_gb"] for i in done if i in stats and stats[i]["peak_gb"] is not None
        ]
        status = "done" if done and not assembled["missing"] else ("partial" if done else "failed")
        info = {
            "model": model,
            "test": test,
            "repo": card.repo,
            "size": card.size,
            "licence": card.licence,
            "status": status,
            "load_s": run["load_s"],
            "segments_total": len(rows),
            "segments_done": len(done),
            "segments_empty": sum(r["empty"] for r in rows),
            "segments_failed": errors,
            "compute_s": round(compute, 2),
            "audio_s": round(audio, 2),
            "compute_per_audio_s": round(compute / audio, 3) if audio else None,
            "peak_gpu_gb": max(peaks) if peaks else None,
            "device": spec.device,
            "missing": assembled["missing"],
            "notes": ([spec.note] if spec.note else []) + run["notes"],
        }
        if status == "failed":
            info["why"] = (
                failed
                or (run["notes"] or list(errors.values()) or ["no segment rendered"])[-1][:800]
            )
        (model_dir / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        infos[test] = info
    return infos
