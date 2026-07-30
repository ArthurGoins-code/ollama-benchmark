# Ollama Coding Benchmark

A single-file, dependency-free Python script that benchmarks multiple
[Ollama](https://ollama.com) models on coding tasks, scoring them on **code
correctness**, **tool-use / agentic planning ability**, **generation speed**,
and **GPU power efficiency** — then ranks them with a composite score to
help you pick the best model for coding work.

## How it works

For every `(model, task)` pair the script:

1. Starts a background GPU sampler that polls `nvidia-smi` for power draw
   while the model is generating.
2. Runs `ollama run <model> "<prompt>" --verbose` and captures:
   - **stdout** — the model's answer (code or a JSON tool-call plan)
   - **stderr** — Ollama's `--verbose` performance stats (load duration,
     prompt-eval / eval token counts, tokens/sec)
3. Evaluates the response:
   - **`python_exec` tasks** (FizzBuzz, word reversal, bug fixing) — extracts
     the code block, checks it compiles, then executes it against a small
     hard-coded test harness in an isolated subprocess. Score = % of test
     assertions passed.
   - **`tool_json` tasks** (bug-fix planning, feature-implementation
     planning) — asks the model to output a JSON array describing the
     sequence of tool calls (`read_file`, `write_file`, `run_tests`, etc.) it
     would make as an autonomous coding agent. Scored heuristically on valid
     JSON, required-tool coverage, and logical call ordering.
4. Records energy used (Joules, integrated from power samples) and derives
   tokens-per-joule as an efficiency metric.

Finally, results are aggregated per model and combined into a **weighted
composite score**, then printed as a leaderboard and saved to JSON/CSV.

### Important limitation: tool-use scoring is a proxy

`ollama run --verbose` is a single-shot text generation call — it does not
actually execute tools or run a real agent loop. There is no way to observe
genuine tool-calling behavior through this CLI. The `tool_json` tasks are
therefore a **proxy**: the model is asked to *describe* the plan of tool
calls it would make, and that plan is scored heuristically. This approximates
agentic planning ability but does not guarantee real-world tool-use
correctness (e.g. it won't catch a model that plans well but produces
malformed tool-call arguments in an actual agent framework).

## Prerequisites

- **Ollama** installed and `ollama serve` running, with the models you want
  to test already pulled (`ollama pull <model>`).
- **`nvidia-smi`** on PATH for power monitoring (NVIDIA GPUs only). If it's
  missing, the script still runs but skips power/efficiency metrics (a
  warning is printed).
- **Python 3.8+** — standard library only, no third-party packages required.

## Usage

```bash
# Benchmark specific models
python3 ollama_coding_benchmark.py --models qwen2.5-coder:7b codellama:13b

# Auto-discover every locally installed model (via `ollama list`) and allow user to pick model(s) to benchmark
python3 ollama_coding_benchmark.py

# List the built-in benchmark tasks
python3 ollama_coding_benchmark.py --list-tasks

# Only run the tool-use / agentic tasks, repeat each 3 times, skip power monitoring
python3 ollama_coding_benchmark.py --models modelA modelB \
    --tasks tool_plan_bugfix agentic_feature_impl --iterations 3 --no-power
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--models MODEL [MODEL ...]` | all installed models | Ollama model names to benchmark (e.g. `qwen2.5-coder:7b`). |
| `--tasks TASK [TASK ...]` | all built-in tasks | Subset of task IDs to run. See `--list-tasks`. |
| `--list-tasks` | — | Print the built-in tasks and exit. |
| `--iterations N` | `1` | Number of times to repeat each task per model. |
| `--timeout SECONDS` | `300` | Per-run timeout. |
| `--output-dir DIR` | `benchmark_results` | Directory to write JSON/CSV reports to. |
| `--power` / `--no-power` | `--power` enabled | Enable/disable GPU power monitoring via `nvidia-smi`. |
| `--gpu-index INDEX` | `0` | GPU index passed to `nvidia-smi`, or `all` to sum across every GPU. |
| `--power-interval SECONDS` | `0.5` | Seconds between power samples. |
| `--no-warmup` | warmup enabled | Skip the warmup run that loads each model into memory before timing starts. |
| `--weight-correctness` | `0.4` | Composite score weight for code correctness. |
| `--weight-tool-use` | `0.2` | Composite score weight for tool-use/agentic planning. |
| `--weight-speed` | `0.2` | Composite score weight for generation speed (tokens/sec, normalized). |
| `--weight-power` | `0.2` | Composite score weight for power efficiency (tokens/joule, normalized). |

## Built-in tasks

| Task ID | Category | Type | What it tests |
|---|---|---|---|
| `fizzbuzz` | code_generation | `python_exec` | Basic algorithm implementation |
| `reverse_words` | code_generation | `python_exec` | String manipulation |
| `bug_fix_palindrome` | debugging | `python_exec` | Finding and fixing a logic bug |
| `tool_plan_bugfix` | tool_use | `tool_json` | Planning tool calls to inspect, fix, and verify a bug |
| `agentic_feature_impl` | agentic | `tool_json` | Planning a multi-step feature change with validation |

## Output

- **Console leaderboard** ranking each model by composite score, with
  correctness, tool-use score, success rate, tokens/sec, average power (W),
  and tokens/joule.
- **`benchmark_results/benchmark_<timestamp>.json`** — full raw results for
  every run (stdout/stderr, parsed stats, power samples summary, eval
  results) plus the per-model aggregates.
- **`benchmark_results/benchmark_<timestamp>_summary.csv`** — a compact
  per-model summary suitable for spreadsheets.

## Notes & tips

- Run `ollama pull <model>` for every model beforehand; the script does not
  pull models automatically.
- The warmup run (enabled by default) loads each model into VRAM before
  timed runs start, so load-time doesn't skew the first task's speed/power
  numbers. Disable with `--no-warmup` if you specifically want to measure
  cold-start behavior.
- Adjust `--weight-*` flags to match what you care about most — e.g. set
  `--weight-power 0.5 --weight-correctness 0.5 --weight-tool-use 0 --weight-speed 0`
  if you only care about correctness vs. power efficiency.
- Power monitoring requires an NVIDIA GPU with `nvidia-smi`. On systems
  without a supported GPU, use `--no-power` to skip it explicitly and avoid
  the startup warning.
