# Ollama Coding Benchmark

A single-file, **dependency-free** Python tool that benchmarks multiple
[Ollama](https://ollama.com) models on realistic coding work and ranks them
with a weighted composite score. It measures four dimensions:

- **Correctness** — does the generated code actually pass its test cases?
- **Agentic / tool-use** — can the model plan and drive a real tool-calling
  loop (read, write, edit, run) to complete a multi-step task?
- **Speed** — tokens generated per second.
- **Power efficiency** — tokens per joule, sampled live from `nvidia-smi`.

## How it works

For every `(model, task)` pair the tool:

1. (Optionally) starts a background GPU sampler that polls `nvidia-smi` for
   power draw while the model generates.
2. Generates a response, either through the **Ollama HTTP API** (default) or
   the `ollama run <model> "<prompt>"` **CLI fallback**, capturing both the
   answer text and Ollama's verbose performance stats (prompt/eval token
   counts, tokens/sec).
3. Evaluates the response:
   - **`code` tasks** (Python / JavaScript / Bash) — extracts the fenced code
     block and runs it against a small, isolated test harness in a
     subprocess. Score = fraction of assertions passed.
   - **`tool` tasks** — two modes (see [Tool-use modes](#tool-use-modes)):
     a one-shot **proxy** JSON plan scored heuristically, or a **native**
     tool-calling agent loop that really invokes read/write/edit/run tools and
     verifies the final file state.
4. Records energy (Joules, integrated from power samples) and derives
   tokens-per-joule.

Results are aggregated per model, combined into a **weighted composite
score**, printed as a leaderboard, and written to JSON/CSV/Markdown.

## Prerequisites

- **Python 3.8+** — standard library only; no third-party packages at runtime.
- **Ollama** with the models you want to test (`ollama pull <model>`). Either
  the HTTP API (default backend) or the `ollama` CLI must be reachable.
- **`nvidia-smi`** on PATH for power monitoring (NVIDIA GPUs). If missing, the
  tool still runs and simply skips the power/efficiency metric.
- **`node`** (JavaScript tasks) and **`bash`** (Bash tasks). Tasks whose
  interpreter is unavailable score 0 and are clearly reported.

## Installation

Run directly — there is nothing to install:

```bash
git clone https://github.com/ArthurGoins-code/model-benchmark.git
cd model-benchmark
python3 ollama_coding_benchmark.py --help
```

Or install it (optional) to get the `ollama-benchmark` command:

```bash
pip install .              # adds the `ollama-benchmark` CLI
ollama-benchmark --version
```

## Usage examples

```bash
# Benchmark a couple of models across every built-in task
python3 ollama_coding_benchmark.py --models qwen2.5-coder:7b llama3.1:8b

# List the built-in task set
python3 ollama_coding_benchmark.py --list-tasks

# Only the Python tasks, three iterations each, via the CLI backend
python3 ollama_coding_benchmark.py --models deepseek-coder:16b \
    --language python --iterations 3 --backend cli

# A real tool-calling agent loop (native mode), up to 8 turns
python3 ollama_coding_benchmark.py --models qwen2.5-coder:32b \
    --task-categories Agent --tool-mode native --max-agent-turns 8

# Resume an interrupted run from its checkpoint
python3 ollama_coding_benchmark.py --models llama3.1:8b --resume

# Compare two finished runs (delta B - A)
python3 ollama_coding_benchmark.py --compare outputs/run_a/report.json outputs/run_b/report.json
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--models M [M ...]`, `-m` | `all` | Model name(s) to benchmark (space-separated); `all` = every installed model. |
| `--task-categories C [C ...]` | all | Only these categories (e.g. `Python JavaScript Agentic`). |
| `--difficulty {easy,medium,hard,all}` | `all` | Only tasks of this difficulty. |
| `--language {python,javascript,bash,all}` | `all` | Only code tasks in this language. |
| `--temperature F` | — | Sampling temperature. |
| `--seed N` | — | Random seed (reproducible sampling). |
| `--num-ctx N` | — | Context window size. |
| `--num-predict N` | — | Max tokens to generate. |
| `--top-k N` / `--top-p F` | — | Sampling parameters. |
| `--backend {api,cli}` | `api` | Ollama HTTP API or `ollama run` CLI fallback. |
| `--host HOST` | `http://127.0.0.1:11434` | Ollama server host (or `$OLLAMA_HOST`). |
| `--timeout SEC` | `300` | Per-request timeout (seconds). |
| `--tool-mode {proxy,native}` | `proxy` | Tool-use mode: one-shot proxy or real agent loop. |
| `--max-agent-turns N` | `6` | Max agent-loop turns (native mode). |
| `--iterations N` | `1` | Independent iterations per model/task. |
| `--workers N` | `1` | Concurrent workers (1 = serial). |
| `--normalize {relative,absolute}` | `relative` | Speed/power normalization basis. |
| `--ref-rate F` / `--ref-tpj F` | — | Absolute references for speed / power. |
| `--thermal-threshold F` | `80.0` | °C at/above which a power drop counts as thermal throttling. |
| `--output-dir DIR` | `outputs` | Directory for reports + checkpoint. |
| `--resume` | off | Resume from an existing checkpoint. |
| `--list-tasks` | off | Print the built-in task set and exit. |
| `--compare A B` | — | Compare two `report.json` files (delta B − A) and exit. |
| `--quiet` / `--verbose` | off | Suppress / add per-run output. |
| `--version` | — | Print version and exit. |

## Built-in tasks

25 tasks across four categories (run `--list-tasks` for the live list):

| ID | Category | Type | Level |
|---|---|---|---|
| `py_reverse_string` | Python | code | easy |
| `py_fizzbuzz` | Python | code | easy |
| `py_is_palindrome` | Python | code | easy |
| `py_count_vowels` | Python | code | easy |
| `py_matrix_transpose` | Python | code | easy |
| `py_binary_search` | Python | code | medium |
| `py_two_sum` | Python | code | medium |
| `py_word_frequency` | Python | code | medium |
| `py_merge_intervals` | Python | code | medium |
| `py_email_extract` | Python | code | medium |
| `py_lru_cache` | Python | code | hard |
| `py_bfs_shortest_path` | Python | code | hard |
| `py_token_bucket` | Python | code | hard |
| `js_group_by` | JavaScript | code | medium |
| `js_string_stats` | JavaScript | code | easy |
| `bash_count_lines` | Bash | code | easy |
| `tool_lookup_weather` | Tool Use | tool | easy |
| `tool_send_email` | Tool Use | tool | medium |
| `tool_data_pipeline` | Tool Use | tool | medium |
| `tool_deploy_pipeline` | Tool Use | tool | hard |
| `native_create_and_run` | Agent | tool | medium |
| `native_fix_buggy` | Agent | tool | hard |
| `native_multi_file` | Agent | tool | hard |
| `native_search_replace` | Agent | tool | medium |
| `native_refactor` | Agent | tool | hard |

## Scoring & composite

- **Correctness** — mean fraction of test assertions passed across `code`
  tasks (0–100).
- **Agentic** — mean `tool`-task score across proxy/native tasks (0–100).
- **Speed** — tokens/sec, normalized (`relative` to the batch max, or
  `absolute` against `--ref-rate`).
- **Power** — tokens/joule, normalized against the batch max or `--ref-tpj`.
- **Thermal throttling** — detected from the power/temperature trace and flagged per
  run and per model, so you know when a speed/power number was being throttled.
- **Composite** — a weighted sum of the four; the leaderboard rank is by
  composite score.

## Tool-use modes

- **`--tool-mode proxy`** (default) — the model *describes* the sequence of
  tool calls it would make as a JSON plan; the plan is scored on valid JSON,
  required-tool coverage, and call ordering. This does **not** run the tools.
- **`--tool-mode native`** — a real tool-calling loop: the model invokes
  `read_file` / `write_file` / `edit` / `run` / `search` against an isolated
  sandbox (up to `--max-agent-turns`) and is scored on whether the final file
  state passes verification. This measures genuine agentic behavior.

## Thermal throttling detection

While sampling power, the tool also records GPU **temperature** and **SM clock**. If,
under continued generation load, power draw falls materially below the run's peak while
the card is at/above the thermal threshold (default **80 °C**, tune with
`--thermal-threshold`), the run is flagged as **thermally throttled** — matching the
"near-max watts dropping without the load decreasing" symptom. A brief cool-down dip at
the end of generation is not flagged (by then the card is already below the threshold).

Thermal status appears everywhere the power metric does:

- per-run `power` object in `report.json` (`thermal_throttled`, `thermal_throttle_events`,
  `thermal_power_drop_pct`, `gpu_temp_max_c`, and the driver's `throttle_reason_flags`),
- `results.csv` columns `gpu_temp_max_c`, `thermal_throttled`, `thermal_throttle_events`,
- the leaderboard (console + Markdown) `Max °C` and `Therm?` columns,
- and a `[THROTTLED]` marker on the live progress line.

If your card throttles at a different point, pass `--thermal-threshold <°C>` —
e.g. lower it (e.g. `--thermal-threshold 75`) if it throttles earlier, or raise it
(e.g. `--thermal-threshold 88`) to reduce false positives.

## Output

Reports are written under `--output-dir` (default `outputs/`):

- `report.json` — full report: `system_info`, config/weights, raw `runs`,
  per-model aggregates (`summary`), and the ranked leaderboard data.
- `results.csv` — flat per-run rows for spreadsheet use (incl. `gpu_temp_max_c`,
  `thermal_throttled`, `thermal_throttle_events`).
- `report.md` — a human-readable Markdown report.
- `benchmark.ckpt.json` — the checkpoint used by `--resume` (safe to delete).

The leaderboard is also printed to the console at the end of each run.

## Resuming & comparing

- **`--resume`** picks up where a previous run in the same `--output-dir`
  left off, reusing already-completed `(model, task, iteration)` results.
- **`--compare A B`** prints a per-model delta table (B − A) across composite,
  correctness, agentic, speed, power, and pass rate.

## Running the tests

The task definitions and evaluation harnesses are covered by a dependency-free
behavioral test suite (no Ollama server required) that runs each task against
a reference solution and also confirms wrong solutions are rejected:

```bash
python3 tests/test_task_definitions.py     # standalone, no dependencies
pip install .[dev] && ruff check .         # lint (as CI does)
```

## Project layout

```
model-benchmark/
├── ollama_coding_benchmark.py   # the whole tool (single file)
├── tests/
│   └── test_task_definitions.py # reference-solution + rejection tests
├── .github/workflows/ci.yml     # CI: compile, lint, task tests
├── pyproject.toml
├── LICENSE                      # MIT
└── README.md
```

## License

[MIT](LICENSE) © Arthur Goins