"""Step 4: the results table (`output/results.md`), the listening page (`output/listen/
index.html`, one player per model and test, with a rating table), and the small server that
saves those ratings into the page itself.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .registry import ALL_MODELS

TEMPLATE = Path(__file__).resolve().parent / "listen.html"
LISTEN = Path(__file__).resolve().parent.parent / "output" / "listen"
RATINGS = re.compile(r'(<script id="ratings" type="application/json">)(.*?)(</script>)', re.S)


def test_title(path: Path) -> str:
    """The first `# comment` of a test file, else its name."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#"):
            return line.strip().lstrip("#").strip()
        if line.strip():
            break
    return path.stem


def _runs(output: Path, tests: list[str]) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = {}
    for test in tests:
        for run in sorted((output / test).glob("*/run.json")):
            info = json.loads(run.read_text(encoding="utf-8"))
            by_model.setdefault(info["model"], []).append(info)
    return by_model


def _summary(infos: list[dict]) -> dict:
    compute = sum(i.get("compute_s") or 0 for i in infos)
    audio = sum(i.get("audio_s") or 0 for i in infos)
    peaks = [i["peak_gpu_gb"] for i in infos if i.get("peak_gpu_gb") is not None]
    return {
        "done": sum(i.get("segments_done", 0) for i in infos),
        "total": sum(i.get("segments_total", 0) - i.get("segments_empty", 0) for i in infos),
        "speed": round(compute / audio, 2) if audio else None,
        "peak": max(peaks) if peaks else None,
        "cpu": any(i.get("device") == "cpu" for i in infos),
        "failed": [f"{i['test']}: {i.get('why', '')}" for i in infos if i["status"] == "failed"],
    }


def write_results(output: Path, tests: list[str]) -> Path:
    rows = []
    for model, infos in _runs(output, tests).items():
        card, s = ALL_MODELS[model], _summary(infos)
        rows.append((s["speed"] if s["speed"] is not None else 1e9, model, card, s))
    rows.sort()
    lines = [
        f"# Results: {', '.join(tests)}",
        "",
        "Speed = compute seconds per second of audio (lower is faster; under 1 is faster than "
        "real time). Segments = rendered / with something to say (sound-only segments a model "
        "has no tag for are left out). Peak GPU = torch's peak allocation, or nvidia-smi's total "
        "for server-based models.",
        "",
        "| Model | Params | Licence | Segments | Speed | Peak GPU | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, model, card, s in rows:
        peak = "CPU" if s["cpu"] else (f"{s['peak']:.1f} GB" if s["peak"] is not None else "-")
        speed = f"{s['speed']:.2f}" if s["speed"] is not None else "-"
        notes = "; ".join(s["failed"])[:200]
        lines.append(
            f"| {model} | {card.size} | {card.licence} | {s['done']}/{s['total']} | "
            f"{speed} | {peak} | {notes} |"
        )
    path = output / "results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_sheet(
    output: Path,
    tests: list[str],
    titles: dict[str, str],
    run_name: str,
    scripts: dict[str, str] | None = None,
) -> Path:
    listen = output / "listen"
    listen.mkdir(parents=True, exist_ok=True)
    runs = _runs(output, tests)
    models = []
    for model, infos in sorted(runs.items(), key=lambda kv: _summary(kv[1])["speed"] or 1e9):
        if not any((listen / model).glob("*.wav")):
            continue
        card, s = ALL_MODELS[model], _summary(infos)
        meta = [card.size, card.licence]
        if s["speed"] is not None:
            meta.append(f"{s['speed']:.2f} s/s")
        if card.baseline:
            meta.append("no cloning")
        models.append([model, " · ".join(meta)])
    clips = []
    for test in tests:
        clips.append([f"{test}.wav", f"{test}: {titles[test]}"])
        scene_files = sorted({p.name for p in listen.glob(f"*/{test}-scene-*.wav")})
        clips += [[name, f"{test}, scene {name.rsplit('-', 1)[1][:-4]}"] for name in scene_files]
    config = {
        "title": "TTS listening sheet",
        "run": run_name,
        "models": models,
        "clips": clips,
        "scripts": scripts or {},
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__CONFIG__", json.dumps(config, ensure_ascii=False).replace("</", "<\\/")
    )
    page = listen / "index.html"
    if page.exists():  # keep the ratings already made
        old = RATINGS.search(page.read_text(encoding="utf-8"))
        if old:
            html = RATINGS.sub(lambda m: m.group(1) + old.group(2) + m.group(3), html, 1)
    page.write_text(html, encoding="utf-8")
    return page


# --- the listening server ---------------------------------------------------------------


def _save_ratings(page: Path, data: dict, lock: threading.Lock) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=1).replace("</", "<\\/")
    with lock:
        html = page.read_text(encoding="utf-8")
        new, n = RATINGS.subn(lambda m: m.group(1) + "\n" + payload + "\n" + m.group(3), html, 1)
        if n != 1:
            raise ValueError("ratings block not found in index.html")
        fd, tmp = tempfile.mkstemp(dir=page.parent, prefix=".index.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new)
        os.replace(tmp, page)


def serve(listen: Path, port: int = 8765) -> None:
    """Serve the listening page; every rating change is written into index.html. Audio is
    served with Range support so the players can seek. Localhost only."""
    page, lock = listen / "index.html", threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(listen), **kwargs)

        def log_message(self, fmt, *args):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            try:
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if not isinstance(data, dict):
                    raise ValueError("expected a JSON object")
                _save_ratings(page, data, lock)
            except Exception as exc:  # noqa: BLE001 - reported to the page
                print(f"save failed: {exc}", flush=True)
                self.send_error(500, str(exc))
                return
            self.send_response(204)
            self.end_headers()

        def do_GET(self):
            rng = self.headers.get("Range")
            path = Path(self.translate_path(self.path))
            m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
            if not m or not path.is_file():
                return super().do_GET()
            size = path.stat().st_size
            start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2) or 0))
            end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}/"
    print(f"Listening page: {url}  (ratings save into {page}; Ctrl+C to stop)", flush=True)
    threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def cli() -> None:
    """`uv run server`: open the listening page so anyone can listen, compare and rate."""
    import argparse

    parser = argparse.ArgumentParser(description=cli.__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not (LISTEN / "index.html").exists():
        raise SystemExit("no listening page yet: run `uv run main.py` first")
    serve(LISTEN, args.port)
