# TTS Benchmark

Render the same short scripts with **33 open text-to-speech models**, then listen to them side
by side and rate them. The scripts are full of the things TTS models usually get wrong: laughs,
moans, slurps, yelps of pain, sneezes, whispers, mumbling, yawns. The question it answers: **which
open model makes cloned characters sound like real people when they aren't just reading words?**

An LLM does the language work: it labels every line (who reads it, emotion, which spans are
sounds) and then writes each model's version of the script in that model's own dialect (its
sound tags like `[laugh]`, `(sighs)` or `<|sfx:cough|>`, its emotion vector, its style prompt).
Every model then runs in its own Python environment, one at a time on the GPU, and you get:

- `output/listen/index.html` (`uv run server`): one player per model and test, the script to
  read along, and a rating table (natural sounding, voicing, other sounds, background, total)
  that saves into the page itself;
- `output/results.md`: speed, peak GPU memory and segments rendered for every model.

## Quick start

Needs Linux, an NVIDIA GPU (tested on an RTX 5090, 32 GB, driver 595), [uv](https://docs.astral.sh/uv/),
git, and a local LLM behind an OpenAI-compatible API (e.g. [LM Studio](https://lmstudio.ai)).

```sh
git clone --recurse-submodules https://github.com/Tejasvedagiri/TTS-Benchmark
cd TTS-Benchmark
cp .env.example .env            # set LLM_MODEL (and HF_TOKEN for the gated models)
uv run main.py --models kokoro,chatterbox_turbo,dia    # a quick first run
uv run main.py                  # every test with every model (hours; resumable)
uv run server                   # listen, read along and rate in the browser
```

A run is resumable: every LLM answer is cached and every rendered segment is kept, so running the
same command again only does what's missing.

## Write your own test

Put a text file in `input/` (e.g. `input/test-11.txt`). One spoken line per line, blank lines
between scenes, the first `#` line is the title shown on the listening page:

```
# Winter soup: mom and son slurping and moaning with delight
Narrator: *A snowy winter evening. Two steaming bowls of tomato soup sit on the table.*
Mom: *She wraps her cold hands around the bowl and sips.* "Mmmmmm. Mmm. Oh, that's warm."
Leo: *A long, loud slurp.* "Mmmmmmm! Oh wow. Ten out of ten."
```

- `Speaker: "speech"`: quoted text is what the speaker says (bare text works too).
- `*asides*`: starred text is read by the Narrator, or as the speaker's thought (the LLM decides).
- Write sounds the way people do ("Ha ha ha!", "Mmmmm", "Ow, ow, ow!", "ACHOO!"): the LLM marks
  them as sounds, and each model gets its own tag for them, or the span is dropped when the model
  can't make that sound.

`input/test-1.txt` ... `test-10.txt` are the included examples: the quick brown fox (a plain
control), winter soup, the mechanic who hits his head on the bonnet, a businessman mumbling his
presentation at the elevator, a whispering sleepover, grandpa laughing at his own joke, a
marathon finish, a sick call, a spider scare and a dad falling asleep at bedtime.

`uv run main.py input/test-11.txt` runs just that test.

## Voices

The cloning models need a short reference clip per speaker. Put yours in `input/voices/`:
`<Speaker>.wav` plus `<Speaker>.txt` with its exact transcript (e.g. `input/voices/mom.wav` +
`mom.txt`; names match case-insensitively). 5-15 s of clean speech works best.

Any speaker without a clip gets a placeholder made with Kokoro-82M (a preset of the speaker's
gender, as the LLM cast them in `output/<test>/cast.json`). The models that can't clone
(kokoro, orpheus, voxtral_tts: presets; maya1: a written voice description) use the same cast.
Edit `cast.json` to change a gender or description; delete it to ask the LLM again.

## Configuration

`.env` (or environment variables):

| Setting | Default | |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:1234/v1` | any OpenAI-compatible server (LM Studio, Ollama, vLLM, llama.cpp, a hosted API) |
| `LLM_MODEL` | the server's default | the model that labels and writes the scripts; a capable 20-30B model works well |
| `LLM_API_KEY` | none | for hosted APIs |
| `LLM_NO_THINK` | off | appends `/no_think` (Qwen3-style thinking models) |
| `HF_HOME` | `~/.cache/huggingface` | where model weights go (roughly 300 GB for every model) |
| `HF_TOKEN` | none | for the gated models (accept their licence on Hugging Face first) |

Gated models: `kyutai/pocket-tts`, `sesame/csm-1b`, `canopylabs/orpheus-3b-0.1-ft` and
`-pretrained`, `neuphonic/neutts-air`, `neuphonic/neutts-nano`, `neuphonic/neucodec`,
`google/t5gemma-2b-2b-ul2` (for T5Gemma-TTS), `Aratako/T5Gemma-TTS-2b-2b`. A model you
haven't accepted fails with a 403 in its `run.json`, and the run moves on.

LM Studio's model is unloaded (`lms unload --all`) before rendering so the voice models get the
GPU; `--keep-llm` skips that. Other flags: `uv run main.py --help`.

## Models

`uv run main.py --list-models` prints them all. Speed and memory from a reference run on an RTX
5090 (79 segments of a 40-line story; speed = compute seconds per second of audio, lower is
faster):

| Model | Params | Licence | Clones | Sound tags | Speed | Peak GPU |
|---|---|---|---|---|---|---|
| styletts2 | 148M | MIT | yes | - | 0.03 | 1.3 GB |
| kokoro | 82M | Apache 2.0 | presets | - | 0.04 | 0.8 GB |
| omnivoice | ~1B | Apache code / NC weights | yes | 2 | 0.11 | 2.3 GB |
| chatterbox_nano | 110M | MIT | yes | 15 | 0.12 | 2.4 GB |
| chatterbox_turbo | 744M | MIT | yes | 19 | 0.15 | 3.3 GB |
| f5tts | 330M | CC-BY-NC | yes | - | 0.17 | 2.1 GB |
| drama3 | 36B MoE (int4) | Fish Audio research NC | yes | open | 0.17 | 31.1 GB\* |
| higgs_v2 | 3B (5.8B total) | research NC | yes | - | 0.19 | 14.1 GB |
| voxtral_tts | 4B | CC-BY-NC | presets | - | 0.20 | 27.0 GB\* |
| pocket_tts | 100M | Apache 2.0 | yes | - | 0.20 | CPU |
| chatterbox | 1.2B | MIT | yes | - | 0.23 | 3.7 GB |
| moss_tts_nano | 100M | Apache 2.0 | yes | - | 0.25 | 0.6 GB |
| higgs_v3 | 4B | research NC | yes | 11 | 0.25 | 29.6 GB\* |
| voxcpm2 | 2B | Apache 2.0 | yes | - | 0.26 | 6.4 GB |
| neutts_nano | 229M | Apache 2.0 | yes | - | 0.29 | 4.1 GB |
| cosyvoice3 | 0.5B | Apache 2.0 | yes | - | 0.31 | 3.6 GB |
| neutts_air | 748M | Apache 2.0 | yes | - | 0.31 | 5.2 GB |
| moss_tts | 8B | Apache 2.0 | yes | pause | 0.37 | 24.4 GB |
| step_audio_editx | 3B | Apache 2.0 | yes | 19 | 0.44 | 3.6 GB |
| zonos | 1.6B | Apache 2.0 | yes | - | 0.45 | 4.2 GB |
| kitten_tts2 | 1.7B | Apache 2.0 | yes | - | 0.47 | 9.0 GB |
| qwen3_tts | 1.7B | Apache 2.0 | yes | - | 0.52 | 4.7 GB |
| orpheus | 3B | Apache 2.0 | presets | 8 | 0.58 | 7.6 GB |
| maya1 | 3B | Apache 2.0 | described | 4 | 0.59 | 7.5 GB |
| fish_s2 | 4B | research NC | yes | open | 0.62 | 20.9 GB |
| indextts2 | 1.5B | Apache 2.0 | yes | - | 0.62 | 7.7 GB |
| llasa_8b | 8B | CC-BY-NC | yes | - | 0.63 | 20.7 GB |
| orpheus_clone | 3B | Apache 2.0 | yes | 8 | 0.63 | 7.6 GB |
| csm_1b | 1B | Apache 2.0 | yes | - | 0.70 | 7.1 GB |
| breeze_tts2 | 3.5B | research NC | yes | open | 0.73 | 8.8 GB |
| t5gemma_tts | 5.3B | CC-BY-NC + Gemma | yes | - | 0.83 | 14.2 GB |
| dia | 1.6B | Apache 2.0 | yes | 9 | 0.88 | 6.2 GB |
| miso_tts | 8B | other (Miso Labs) | yes | - | 1.23 | 19.1 GB |

\* Served by vLLM-Omni / SGLang-Omni: nvidia-smi's total, including the server's up-front
reservation and ~1 GB for the desktop; the model itself needs less. "NC" = non-commercial:
check each model's licence before using its output.

Models without tags still get the labels another way where they can: zonos an emotion vector,
chatterbox exaggeration, indextts2 / voxcpm2 / cosyvoice3 / breeze_tts2 a written emotion or
style, higgs_v3 an emotion token.

## Output

```
output/
  <test>/direction.jsonl          the LLM's labels (who reads what, emotion, sound spans)
  <test>/cast.json                gender, age and voice description per speaker
  <test>/voices/                  Kokoro placeholder clips (+ transcripts)
  <test>/<model>/<test>.engine.txt    what the model was given, readable (.engine.jsonl: exact)
  <test>/<model>/segments/        one wav per segment; run.json, segment-stats.json
  listen/index.html               the listening and rating page; listen/<model>/<test>.wav
  results.md                      the table above, for your run
  logs/<model>.txt                each model's install and render log
```

## How it works

1. **Label** (`bench/direction.py`): the script is split in code (speech, asides); the LLM labels
   each piece and lists the spans that spell out a sound ("Mmmmm" -> moan, "Ow, ow!" -> whimper,
   keep the words). Every span must be found in the text; a line that doesn't check out gets
   plain labels.
2. **Engine scripts** (`bench/engine_scripts.py`): for each model, the LLM rewrites the labelled
   segments using that model's card (`bench/registry.py`): its tag list or tag style, its fields.
   Code checks every answer (same words, only known tags, fields in range), asks again once,
   then falls back to plain text. Models with no tags or text fields need no LLM call.
3. **Render** (`bench/render.py`): each model gets `envs/<model>/` (uv, its own Python and pins,
   CUDA 12.8 torch) on first use; `bench/runners/run_<model>.py` loads it once and renders every
   test's segments. Stalls are killed, failures recorded, and the run moves on.
4. **Report** (`bench/report.py`, `bench/assemble.py`): segments are joined with fixed pauses into
   one file per model and test; the results table and the listening page are written.

To add a model, see [AGENTS.md](AGENTS.md).
