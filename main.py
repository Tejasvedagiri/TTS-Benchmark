"""TTS benchmark: render the same scripts with every open TTS model, side by side.

    uv run main.py                          # every input/*.txt with every model
    uv run main.py input/test-3.txt         # just this test
    uv run main.py --models kokoro,dia      # just these models
    uv run main.py --scripts-only           # LLM steps only: labels + each engine's script
    uv run main.py --render-only            # skip the LLM steps (engine scripts already written)
    uv run server                           # open the listening page (ratings save into it)
    uv run main.py --list-models

Steps, for each test (`input/<test>.txt`):
  1. the LLM labels the script (who reads each piece, emotion, delivery, which spans are sounds)
     and describes the cast -> output/<test>/direction.jsonl, cast.json
  2. the LLM writes each model's version in its own dialect (tags, emotion fields)
     -> output/<test>/<model>/<test>.engine.txt (+ .engine.jsonl)
  3. the LLM is unloaded (LM Studio), speakers without a clip in input/voices/ get a Kokoro
     placeholder, and each model renders every test in its own environment (envs/<model>/)
  4. output/results.md (speed, memory, segments) and output/listen/index.html (listen + rate)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from bench import direction, engine_scripts, report, voices
from bench.llm import LLM, LLMError, load_dotenv, unload_lm_studio
from bench.registry import ALL_MODELS
from bench.render import SPECS, render_model
from bench.script import ScriptError, parse_script, speakers, write_direction

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _natural(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.stem)]


def _tests(args) -> list[Path]:
    paths = [Path(p) for p in args.tests] or sorted(INPUT.glob("*.txt"), key=_natural)
    missing = [str(p) for p in paths if not p.exists()]
    if missing or not paths:
        sys.exit(f"no test scripts: {', '.join(missing) or f'put one in {INPUT}/<name>.txt'}")
    return paths


def _models(args) -> list[str]:
    if args.models == "all":
        return list(SPECS)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in SPECS]
    if unknown:
        sys.exit(f"unknown model(s): {', '.join(unknown)} (see --list-models)")
    return models


def list_models() -> None:
    print(f"{'model':18} {'params':28} {'licence':26} clones  tags")
    for model in SPECS:
        card = ALL_MODELS[model]
        tags = "open" if card.tags == "open" else (len(card.tags) if card.tags else "-")
        print(
            f"{model:18} {card.size:28} {card.licence:26} "
            f"{'no' if card.baseline else 'yes':7} {tags}"
        )


def write_scripts(test_path: Path, models: list[str], llm: LLM) -> dict:
    """Steps 1-2 for one test; returns its voices plan."""
    test = test_path.stem
    test_dir = OUTPUT / test
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        raws = parse_script(test_path)
    except ScriptError as exc:
        sys.exit(str(exc))
    log(f"{test}: {len(raws)} lines, speakers {', '.join(speakers(raws))}")
    cast = direction.cast_voices(raws, speakers(raws), llm, test_dir / "cast.json", log)
    lines = direction.direct(raws, llm, cast, test_dir / "direction-cache.json", log)
    write_direction(test_dir / "direction.jsonl", lines)
    plan = voices.plan_voices(cast, test_dir / "voices")
    for model in models:
        card = ALL_MODELS[model]
        model_dir = test_dir / model
        results = engine_scripts.write_for_model(card, lines, llm, model_dir, log)
        engine_scripts.write_engine_script(card, lines, results, plan, model_dir, test)
    log(f"{test}: wrote {len(models)} engine scripts in output/{test}/<model>/{test}.engine.txt")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tests", nargs="*", help="test scripts (default: every input/*.txt)")
    parser.add_argument("--models", default="all", help="'all' or a comma-separated list")
    parser.add_argument("--scripts-only", action="store_true", help="stop after the LLM steps")
    parser.add_argument("--render-only", action="store_true", help="skip the LLM steps")
    parser.add_argument("--listen", action="store_true", help="same as `uv run server`")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--llm-url",
        default="",
        help="OpenAI-compatible base URL "
        "(default: LLM_BASE_URL, else LM Studio's http://localhost:1234/v1)",
    )
    parser.add_argument("--llm-model", default="", help="default: LLM_MODEL")
    parser.add_argument(
        "--keep-llm", action="store_true", help="don't unload LM Studio's model before rendering"
    )
    args = parser.parse_args()
    load_dotenv()

    if args.list_models:
        list_models()
        return
    if args.listen:
        if not (report.LISTEN / "index.html").exists():
            sys.exit("no listening page yet: run the benchmark first")
        report.serve(report.LISTEN, args.port)
        return

    tests, models = _tests(args), _models(args)
    names = [t.stem for t in tests]
    if not args.render_only:
        llm = LLM.from_env(args.llm_url, args.llm_model)
        log(f"LLM: {llm.describe()}")
        try:
            llm.check()
        except LLMError as exc:
            sys.exit(
                f"the LLM isn't answering ({exc}). Start LM Studio (or any OpenAI-compatible "
                "server) and set LLM_BASE_URL / LLM_MODEL in .env, or pass --llm-url."
            )
        plans = {t.stem: write_scripts(t, models, llm) for t in tests}
        if args.scripts_only:
            return
        if not args.keep_llm:
            unload_lm_studio(log)
    else:
        plans = {}
        for t in tests:
            cast_path = OUTPUT / t.stem / "cast.json"
            if not cast_path.exists():
                sys.exit(f"{t.stem} has no engine scripts yet: run without --render-only first")
            plans[t.stem] = voices.plan_voices(
                json.loads(cast_path.read_text(encoding="utf-8")), OUTPUT / t.stem / "voices"
            )

    voices.make_missing_clips(plans, OUTPUT, log)
    logs = OUTPUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for model in models:
        missing = [n for n in names if not (OUTPUT / n / model / f"{n}.engine.jsonl").exists()]
        if missing:
            log(f"{model}: no engine script for {', '.join(missing)} (run without --render-only)")
            continue
        log(f"{model} ...")
        with (logs / f"{model}.txt").open("a", encoding="utf-8") as fh:
            fh.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {model}: {', '.join(names)}\n")

            def model_log(msg: str, _fh=fh) -> None:
                _fh.write(msg + "\n")
                _fh.flush()

            infos = render_model(model, names, OUTPUT, OUTPUT / "listen", model_log)
        done = sum(i["segments_done"] for i in infos.values())
        total = sum(i["segments_total"] - i["segments_empty"] for i in infos.values())
        failed = [
            f"{t}: {i.get('why', '')[:160]}" for t, i in infos.items() if i["status"] == "failed"
        ]
        log(f"{model}: {done}/{total} segments" + (f"; failed {failed[0]}" if failed else ""))

    # the report and the page cover every test rendered so far, not just this run's
    every = sorted((p.parent for p in OUTPUT.glob("*/direction.jsonl")), key=_natural)
    titles = {
        d.name: report.test_title(INPUT / f"{d.name}.txt")
        if (INPUT / f"{d.name}.txt").exists()
        else d.name
        for d in every
    }
    results = report.write_results(OUTPUT, list(titles))
    scripts = {
        name: (INPUT / f"{name}.txt").read_text(encoding="utf-8")
        for name in titles
        if (INPUT / f"{name}.txt").exists()
    }
    page = report.write_sheet(OUTPUT, list(titles), titles, "output", scripts)
    log(
        f"results: {results.relative_to(ROOT)}; listen and rate: uv run server "
        f"({page.relative_to(ROOT)})"
    )


if __name__ == "__main__":
    main()
