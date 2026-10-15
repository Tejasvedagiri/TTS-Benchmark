# AGENTS.md

Guide for coding agents (and humans) working on this repository. What it does and how to use it:
[README.md](README.md).

## What this is

`uv run main.py` renders the test scripts in `input/*.txt` with every open TTS model in
`bench/registry.py`, so they can be compared by ear (`output/listen/index.html`) and by numbers
(`output/results.md`).

```
input/<test>.txt -> parse (code) -> labels + cast (LLM) -> <test>.engine.txt per model (LLM)
  -> unload LLM -> reference clips (yours, or Kokoro) -> render, one model at a time, each in
  envs/<model>/ -> assemble -> results.md + listen/index.html
```

## Commands

```sh
uv run main.py                               # everything; rerun = resume
uv run main.py input/test-3.txt --models kokoro,dia
uv run main.py --scripts-only                # LLM steps only (no GPU models)
uv run main.py --render-only                 # rendering only (engine scripts exist)
uv run server                                # serve the listening page (bench.report:cli)
uv run main.py --list-models
uvx ruff check . && uvx ruff format --check .
```

Every change must leave `ruff check` and `ruff format --check` clean (line length 100).

## Layout

```
main.py                 the only entry point: CLI, steps in order, logging
bench/
  script.py             test-file format: parse_script() -> RawLine/Piece; Line/Segment/Vocal;
                        direction.jsonl read/write
  llm.py                OpenAI-compatible client (stdlib), .env loading, `lms unload --all`
  direction.py          step 1: LLM labels (apply_labels checks them) + cast.json
  engine_scripts.py     step 2: per-model engine scripts (validate_item, retries, fallback,
                        code-mapped knobs for chatterbox / zonos / styletts2)
  registry.py           ModelCard per model: tags, tag notes, style fields, size, licence
  render.py             SPECS (install recipe per model), ensure_env(), run_runner(),
                        render_model() (all tests of one model in one runner process)
  voices.py             reference clips: input/voices/<speaker>.wav+txt, else Kokoro placeholders
  assemble.py           segments -> listen/<model>/<test>.wav with fixed pauses
  report.py             results.md, listen/index.html (from listen.html), the `uv run server` server
  listen.html           the listening page template (ratings saved in the page itself)
  runners/
    _base.py            the loop every runner shares (resume, atomic writes, JSON events)
    _presets.py         preset voices by gender/slot (kokoro, orpheus, voxtral_tts)
    _vllm_omni.py       start a vLLM-Omni / SGLang-Omni server and post /v1/audio/speech
    _espeak.py, _npp.py, _neutts.py   small shared fixes (espeak library, NPP preload, NeuTTS)
    run_<model>.py      load(device) + synth(engine, row) for one model
input/                  test-*.txt (examples) and voices/ (optional reference clips)
vendor/breeze-tts       git submodule: Breeze-TTS-2's official inference code
envs/, output/, archive/   generated or local only (git-ignored)
```

## Design rules

- **Code splits, the LLM labels.** The parser decides what is speech and what is an aside; the
  LLM only labels pieces and points at sound spans. Never add regex rules that guess sounds,
  tags or emotions from the words: that's the LLM's job, and code only checks its answers
  (spans must be in the text, words must survive, tags must be on the model's list).
- **Every LLM answer is checked and has a fallback.** Labels: `apply_labels` or plain labels.
  Engine scripts: `validate_item`, one retry alone, then plain text with tags dropped. A run
  never stops because the LLM said something odd; the fallback is logged and noted in the row.
- **One model in GPU memory at a time.** The LLM is unloaded before rendering; each runner is
  its own process and exits before the next model starts.
- **Each model in its own environment** (`envs/<model>/`, built by `ensure_env` from its
  `Spec`). Never install a model's packages into the root project; the root needs only numpy
  and soundfile. Runners import nothing from `bench` (they run in another Python): only
  `_base.py` and the other `_*.py` helpers next to them.
- **Everything resumes.** LLM answers are cached (`direction-cache.json`, `engine-cache.json`,
  keyed by text + card + prompt + LLM model); a segment is skipped when its wav exists and was
  rendered from the same engine-script row (`input` hash in segment-stats.json); an env is
  rebuilt only when its spec changes (`.ready` stamp).
- **Never let a model run unbounded.** LLM calls carry `max_tokens` (thinking counts against
  it); runners are killed after `STALL_S` without progress, `LOAD_S` while loading, `MODEL_S`
  overall.
- **Failures are data.** A model that can't install or render gets a `run.json` with
  `status: failed` and `why`, and the run moves on.

## Adding a model

1. **Card** in `bench/registry.py`: repo, size, licence, `ref` (clip / clip + transcript),
   `needs_transcript`, `tags` (None, a list in the model's own syntax, or "open" + `tag_notes`
   and a sample in `engine_scripts.OPEN_TAG_SAMPLE`), `style_keys` for extra fields,
   `baseline=True` if it can't clone, `speaker_prefix` if every segment needs fixed text.
2. **Spec** in `bench/render.py` `SPECS`: Python version, packages, overrides (torch pins that
   work on the GPU), or a `repo` to clone (`repo_install`: requirements / package / sync /
   none). Order in `SPECS` = run order (small and fast first).
3. **Runner** `bench/runners/run_<id>.py`: `load(device)` returns anything; `synth(engine,
   row)` returns `(float32 mono array, sample_rate)`; end with `main(load, synth)`. Use
   `clip_path(row)` / `ref_text(row)` for cloning, `_presets.preset(model, row)` for presets,
   `row["voice_description"]` for described voices, and the card's `style_keys` from `row`.
4. Test it: `uv run main.py input/test-2.txt --models <id>`, then listen to
   `output/listen/<id>/test-2.wav` and read `output/logs/<id>.txt`.
5. Add it to the README's model table.

## Gotchas (all hit on Linux with an RTX 5090; keep them)

- RTX 50-series (sm_120) needs CUDA 12.8 torch >= 2.7: `--torch-backend cu128`, default
  overrides `TORCH_28`. torch 2.8 needs `torchao==0.13.0` (0.14+ wants torch 2.9).
- uv's `--torch-backend` looks for torch-adjacent packages (torchtune) on PyTorch's index only:
  models needing torchtune (NeuTTS, XCodec2 / Llasa, Miso, T5Gemma) use `torch_backend="index"`.
- vLLM-Omni / SGLang-Omni models JIT-compile kernels: `_vllm_omni.py` builds a CUDA_HOME from the
  env's NVIDIA cu13 wheels (`envs/<model>/cuda`, with `lib64` links), and every CUDA compiler
  wheel is pinned to torch's 13.2 (`CUDA_132`); nvcc / nvvm 13.4 emitted PTX the driver rejected.
- Higgs 3 must not get a global `--gpu-memory-utilization` (its deploy config splits the GPU
  between two stages; a global flag OOM'd the codec). Drama 3 needs `max_new_tokens 600` and
  `max_running_requests 4` on 32 GB, and rejects `seed`.
- Chatterbox keeps the package set it first rendered with (`CHATTERBOX_SPEC`): under the date
  cutoff its s3tokenizer got float64 audio against float32 mel filters and every segment failed.
- Windows only (`WINDOWS` in render.py): Smart App Control blocks freshly released native wheels
  (`DLL load failed ... Application Control policy`), hence the older `EXCLUDE_NEWER` cutoff and
  the tokenizers / numpy / transformers caps there.
- Thinking LLMs spend most of `max_tokens` on hidden reasoning: too low a cap returns empty or
  truncated JSON (`LLMError` says `finish_reason` and reasoning tokens). LM Studio's
  `enable_thinking: false` really turns it off; servers that reject the field are retried
  without it.
- `step_audio_editx` can't render a segment that is only a sound tag (it edits sounds into
  words), so a sound-only line is left out for it.
- Runner files are named `run_<model>.py` on purpose: `kokoro.py` / `dia.py` shadowed the
  packages they import.
