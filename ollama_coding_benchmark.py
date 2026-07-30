#!/usr/bin/env python3
"""
ollama_coding_benchmark.py
===========================

Benchmarks multiple Ollama models on coding tasks using the CLI:

    ollama run <model> "<prompt>" --verbose

For every (model, task) pair the script:
  1. Starts a background GPU power sampler (`nvidia-smi --query-gpu=power.draw`)
     while the model is generating.
  2. Runs `ollama run <model> "<prompt>" --verbose` and captures stdout
     (the model's answer) and stderr (Ollama's `--verbose` performance stats:
     total/load/prompt-eval/eval durations and token rates).
  3. Evaluates the generated code:
       - "python_exec" tasks: extracts the code, checks it compiles, then
         executes it against a small hard-coded test harness in a subprocess.
       - "tool_json" tasks: asks the model to plan a sequence of tool calls
         (agentic / tool-use proxy task) and scores the JSON plan it returns
         for validity, tool coverage, and logical ordering.
  4. Records energy used (Joules, integrated from the power samples) and
     tokens-per-joule as an efficiency metric.

Finally it aggregates results per model (correctness, tool-use score,
generation speed, power efficiency) into a single weighted composite score
and prints/saves a leaderboard.

IMPORTANT LIMITATION
---------------------
`ollama run --verbose` is a single-shot text generation call - it does not
actually execute tools or run an agent loop. There is no way to observe real
tool-calling behavior through this CLI. The "tool_json" tasks are therefore a
*proxy*: the model is asked to describe, as structured JSON, the plan of tool
calls it would make. This is scored heuristically (valid JSON, correct tool
names, sensible ordering) as an approximation of agentic planning ability,
not a guarantee of real-world tool-use correctness.

PREREQUISITES
-------------
- `ollama` CLI installed and `ollama serve` running, with the models you
  want to test already pulled (`ollama pull <model>`).
- `nvidia-smi` available on PATH for power monitoring (NVIDIA GPUs only).
  If missing, the script still runs but skips power/efficiency metrics.
- Python 3.8+ (standard library only, no third-party dependencies).

USAGE
-----
    # Benchmark specific models
    python3 ollama_coding_benchmark.py --models qwen2.5-coder:7b codellama:13b

    # Auto-discover every locally installed model and benchmark all of them
    python3 ollama_coding_benchmark.py

    # List the built-in benchmark tasks
    python3 ollama_coding_benchmark.py --list-tasks

    # Only run the tool-use / agentic tasks, repeat each 3 times, skip power
    python3 ollama_coding_benchmark.py --models modelA modelB \\
        --tasks tool_plan_bugfix agentic_feature_impl --iterations 3 --no-power
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ======================================================================
# Task definitions
# ======================================================================

@dataclass
class Task:
    id: str
    category: str
    eval_type: str  # "python_exec" | "tool_json"
    prompt: str
    test_code: Optional[str] = None          # used when eval_type == "python_exec"
    required_tools: Optional[List[str]] = None  # used when eval_type == "tool_json"
    timeout: int = 120


TASKS: List[Task] = [
    Task(
        id="fizzbuzz",
        category="code_generation",
        eval_type="python_exec",
        prompt=(
            "Write a single Python function named `fizzbuzz(n)` that returns a list "
            "of strings for numbers 1..n following the classic FizzBuzz rules "
            "(multiples of 3 -> 'Fizz', multiples of 5 -> 'Buzz', multiples of both "
            "-> 'FizzBuzz', otherwise the number as a string). "
            "Respond with ONLY the code in a single ```python fenced code block, "
            "no explanation."
        ),
        test_code="""
check(fizzbuzz(15) == ['1','2','Fizz','4','Buzz','Fizz','7','8','Fizz','Buzz','11','Fizz','13','14','FizzBuzz'],
      "fizzbuzz(15) full sequence")
check(fizzbuzz(3)[-1] == 'Fizz', "fizzbuzz(3) ends with Fizz")
check(fizzbuzz(5)[-1] == 'Buzz', "fizzbuzz(5) ends with Buzz")
""",
    ),
    Task(
        id="reverse_words",
        category="code_generation",
        eval_type="python_exec",
        prompt=(
            "Write a Python function named `reverse_words(sentence)` that takes a "
            "string and returns a new string with the order of words reversed, "
            "collapsing any repeated whitespace to single spaces and trimming "
            "leading/trailing whitespace. Respond with ONLY the code in a single "
            "```python fenced code block, no explanation."
        ),
        test_code="""
check(reverse_words("the sky is blue") == "blue is sky the", "basic reversal")
check(reverse_words("  hello   world  ") == "world hello", "extra whitespace handling")
check(reverse_words("one") == "one", "single word")
""",
    ),
    Task(
        id="bug_fix_palindrome",
        category="debugging",
        eval_type="python_exec",
        prompt=(
            "The following Python function is supposed to check whether a string "
            "is a palindrome (ignoring case, spaces, and punctuation) but it has a "
            "bug:\n\n"
            "```python\n"
            "def is_palindrome(s):\n"
            "    s = s.lower()\n"
            "    return s == s[::-1]\n"
            "```\n\n"
            "Fix the function so it correctly ignores spaces and punctuation. "
            "Respond with ONLY the corrected function, named `is_palindrome`, in a "
            "single ```python fenced code block, no explanation."
        ),
        test_code="""
check(is_palindrome("A man, a plan, a canal: Panama") == True, "classic palindrome phrase")
check(is_palindrome("Not a palindrome") == False, "non palindrome")
check(is_palindrome("") == True, "empty string")
""",
    ),
    Task(
        id="tool_plan_bugfix",
        category="tool_use",
        eval_type="tool_json",
        prompt=(
            "You are an autonomous coding agent with access to the following "
            "tools:\n"
            "- read_file(path: str) -> str\n"
            "- write_file(path: str, content: str) -> None\n"
            "- run_tests() -> str\n\n"
            "Task: the file `utils.py` contains a function `add(a, b)` that "
            "incorrectly subtracts instead of adding. Produce a step-by-step plan, "
            "as a JSON array of tool call objects, to inspect the file, fix the "
            "bug, and verify the fix. Each object must have exactly the fields "
            '"tool" and "args". Respond with ONLY the JSON array - no explanation, '
            "no markdown fences."
        ),
        required_tools=["read_file", "write_file", "run_tests"],
    ),
    Task(
        id="agentic_feature_impl",
        category="agentic",
        eval_type="tool_json",
        prompt=(
            "You are an autonomous coding agent with access to the following "
            "tools:\n"
            "- search_code(query: str) -> str\n"
            "- read_file(path: str) -> str\n"
            "- write_file(path: str, content: str) -> None\n"
            "- run_tests() -> str\n\n"
            "Task: add input validation to an existing `parse_config(path)` "
            "function so it raises a clear error on missing required keys, "
            "without breaking existing behavior. Produce a step-by-step plan, as "
            "a JSON array of tool call objects, describing how you would locate "
            "the relevant code, make the change, and confirm nothing else broke. "
            'Each object must have exactly the fields "tool" and "args". Respond '
            "with ONLY the JSON array - no explanation, no markdown fences."
        ),
        required_tools=["search_code", "write_file", "run_tests"],
    ),
]

TASKS_BY_ID = {t.id: t for t in TASKS}


# ======================================================================
# GPU power monitoring
# ======================================================================

class PowerMonitor:
    """Samples `nvidia-smi` power draw on a background thread."""

    def __init__(self, gpu_index: str = "0", interval: float = 0.5):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: List[tuple] = []  # (timestamp, watts)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample_power(self) -> float:
        cmd = ["nvidia-smi", "--query-gpu=power.draw",
               "--format=csv,noheader,nounits"]
        if self.gpu_index != "all":
            cmd += ["-i", str(self.gpu_index)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
        watts = 0.0
        for line in out.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if parts:
                try:
                    watts += float(parts[0])
                except ValueError:
                    continue
        return watts

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                watts = self._sample_power()
                self.samples.append((time.time(), watts))
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        self.samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def energy_joules(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            dt = t1 - t0
            energy += (p0 + p1) / 2.0 * dt
        return energy

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {"available": False}
        watts = [w for _, w in self.samples]
        duration = self.samples[-1][0] - self.samples[0][0]
        result = {
            "available": True,
            "avg_power_w": statistics.mean(watts),
            "max_power_w": max(watts),
            "min_power_w": min(watts),
            "samples": len(watts),
            "duration_s": duration,
            "energy_j": self.energy_joules(),
        }
        return result


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


# ======================================================================
# System information (CPU / RAM / GPU / OS / Ollama version)
# ======================================================================

def get_cpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "model": None,
        "architecture": platform.machine(),
        "logical_cores": os.cpu_count(),
    }
    model_name = None
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        model_name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    elif platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            model_name = out.stdout.strip() or None
        except Exception:
            pass
    info["model"] = model_name or platform.processor() or "unknown"
    return info


def get_memory_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"total_gb": None}
    system = platform.system()
    if system == "Linux":
        try:
            meminfo = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    key, _, value = line.partition(":")
                    meminfo[key.strip()] = value.strip()
            total_kb = meminfo.get("MemTotal", "").split()[0]
            info["total_gb"] = round(int(total_kb) / (1024 ** 2), 2)
        except Exception:
            pass
    elif system == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            info["total_gb"] = round(int(out.stdout.strip()) / (1024 ** 3), 2)
        except Exception:
            pass
    return info


def get_gpu_info() -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    if not nvidia_smi_available():
        return gpus
    try:
        cmd = ["nvidia-smi",
               "--query-gpu=index,name,memory.total,driver_version,compute_cap",
               "--format=csv,noheader"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total": parts[2],
                    "driver_version": parts[3],
                    "compute_capability": parts[4],
                })
    except Exception:
        pass
    return gpus


def get_ollama_version() -> Optional[str]:
    try:
        out = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10)
        text = (out.stdout or out.stderr).strip()
        return text or None
    except Exception:
        return None


def get_system_info() -> Dict[str, Any]:
    """Collects CPU, RAM, GPU, OS, Python and Ollama version info for the report."""
    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpus": get_gpu_info(),
        "ollama_version": get_ollama_version(),
    }


def list_installed_models() -> List[str]:
    """Get list of installed Ollama models."""
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
    except Exception:
        return []
    lines = proc.stdout.strip().splitlines()
    models = []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def prompt_model_selection(models: List[str]) -> List[str]:
    """Present installed models to user and let them select which to benchmark."""
    if not models:
        print("No models installed. Run 'ollama pull <model>' first.")
        return []

    print("\nInstalled Ollama models:")
    print("-" * 40)
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print(f"  {len(models) + 1}. All models")
    print(f"  {len(models) + 2}. Custom (comma-separated names)")
    print("-" * 40)

    choice = input("\nSelect models (1-{}): ".format(len(models) + 2)).strip()

    if choice == str(len(models) + 1):
        print(f"Using all {len(models)} models.")
        return models
    elif choice == str(len(models) + 2):
        custom = input("Enter model names separated by commas: ").strip()
        if custom:
            return [m.strip() for m in custom.split(",") if m.strip()]
        return models
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return [models[idx]]
        except ValueError:
            pass
        return models


# ======================================================================
# Parsing `ollama run --verbose` performance stats
# ======================================================================

_DURATION_RE = re.compile(r"([\d.]+)(h|ms|µs|us|ns|m|s)")
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "µs": 1e-6, "us": 1e-6, "ns": 1e-9}


def parse_go_duration(text: str) -> float:
    """Parse Go-style duration strings like '2.607021208s' or '45.29ms'."""
    total = 0.0
    for value, unit in _DURATION_RE.findall(text):
        total += float(value) * _UNIT_SECONDS[unit]
    return total


_VERBOSE_PATTERNS = {
    "total_duration_s": r"total duration:\s+(\S+)",
    "load_duration_s": r"load duration:\s+(\S+)",
    "prompt_eval_count": r"prompt eval count:\s+(\d+)",
    "prompt_eval_duration_s": r"prompt eval duration:\s+(\S+)",
    "prompt_eval_rate": r"prompt eval rate:\s+([\d.]+)",
    "eval_count": r"(?<!prompt )eval count:\s+(\d+)",
    "eval_duration_s": r"(?<!prompt )eval duration:\s+(\S+)",
    "eval_rate": r"(?<!prompt )eval rate:\s+([\d.]+)",
}


def parse_verbose_stats(text: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for key, pattern in _VERBOSE_PATTERNS.items():
        m = re.search(pattern, text, re.MULTILINE)
        if not m:
            continue
        value = m.group(1)
        if key.endswith("_duration_s"):
            stats[key] = parse_go_duration(value)
        elif key in ("prompt_eval_count", "eval_count"):
            stats[key] = int(value)
        else:
            stats[key] = float(value)
    return stats


# ======================================================================
# Code extraction & evaluation
# ======================================================================

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_blocks(text: str) -> List[str]:
    return [block.strip() for block in _CODE_BLOCK_RE.findall(text)]


def evaluate_python_exec(response_text: str, test_code: str, timeout: int = 15) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "has_code": False,
        "syntax_valid": False,
        "tests_passed": 0,
        "tests_total": 0,
        "score": 0.0,
        "error": None,
    }

    blocks = extract_code_blocks(response_text)
    code = "\n\n".join(blocks) if blocks else response_text.strip()
    result["has_code"] = bool(code.strip())
    if not result["has_code"]:
        result["error"] = "No code found in response"
        return result

    try:
        compile(code, "<generated>", "exec")
        result["syntax_valid"] = True
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e}"
        result["score"] = 5.0  # small credit for attempting
        return result

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "solution.py").write_text(code)
        harness = textwrap.dedent(f"""
            import sys, json
            sys.path.insert(0, {str(tmp)!r})
            results = {{"passed": 0, "total": 0, "failures": []}}

            def check(cond, name):
                results["total"] += 1
                if cond:
                    results["passed"] += 1
                else:
                    results["failures"].append(name)

            try:
                from solution import *
            except Exception as e:
                results["failures"].append("import error: " + str(e))
                results["total"] = 1
            else:
                try:
{textwrap.indent(test_code.strip(), " " * 20)}
                except Exception as e:
                    results["failures"].append("exception: " + str(e))
                    results["total"] += 1

            print("__RESULT__" + json.dumps(results))
        """)
        harness_path = tmp / "run_tests.py"
        harness_path.write_text(harness)

        try:
            proc = subprocess.run(
                [sys.executable, str(harness_path)],
                capture_output=True, text=True, timeout=timeout, cwd=str(tmp),
            )
            marker = "__RESULT__"
            idx = proc.stdout.find(marker)
            if idx != -1:
                payload = json.loads(proc.stdout[idx + len(marker):].strip().splitlines()[0])
                result["tests_passed"] = payload.get("passed", 0)
                result["tests_total"] = payload.get("total", 0)
                if payload.get("failures"):
                    result["error"] = "; ".join(payload["failures"])[:500]
            else:
                result["error"] = (proc.stderr or "Unknown test harness failure")[:500]
        except subprocess.TimeoutExpired:
            result["error"] = "Execution timed out"

    if result["tests_total"]:
        pass_ratio = result["tests_passed"] / result["tests_total"]
        result["score"] = round(20 + 80 * pass_ratio, 2)
    else:
        result["score"] = 30.0  # syntax valid but no tests could run
    return result


def evaluate_tool_json(response_text: str, required_tools: Optional[List[str]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "valid_json": False,
        "num_calls": 0,
        "tools_used": [],
        "matched_required": 0,
        "required_total": len(required_tools or []),
        "order_bonus": False,
        "args_bonus": False,
        "score": 0.0,
        "error": None,
    }

    text = response_text.strip()
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            try:
                data = json.loads(bracket_match.group(0))
            except json.JSONDecodeError as e:
                result["error"] = f"JSON parse failed: {e}"
        else:
            result["error"] = "No JSON array found in response"

    if data is None:
        return result

    result["valid_json"] = True
    if not isinstance(data, list):
        data = [data]

    calls = [c for c in data if isinstance(c, dict) and "tool" in c]
    result["num_calls"] = len(calls)
    tools_used = [c.get("tool") for c in calls]
    result["tools_used"] = tools_used

    # Check if all calls have "args" field (properly formed tool calls)
    calls_with_args = [c for c in calls if "args" in c]
    result["args_bonus"] = len(calls_with_args) == len(calls) if calls else False

    required = required_tools or []
    matched = [t for t in required if t in tools_used]
    result["matched_required"] = len(matched)

    # Improved scoring: more granular and forgiving
    score = 20.0  # base score for attempting
    
    # Valid JSON array: +10
    if result["valid_json"]:
        score += 10.0
    
    # Has tool calls: +15
    if calls:
        score += 15.0
    
    # All calls have args field: +5
    if result["args_bonus"]:
        score += 5.0
    
    # Required tools coverage: +35 (was 40, more forgiving)
    if required:
        score += 35.0 * (len(matched) / len(required))
    elif calls:
        score += 35.0
    
    # Tool order bonus: +15 (kept as is)
    if required and all(t in tools_used for t in required):
        indices = [tools_used.index(t) for t in required]
        if indices == sorted(indices):
            result["order_bonus"] = True
            score += 15.0
    elif required and len(matched) > 0:
        # Partial order bonus: give credit for partial ordering
        partial_order = sum(1 for i in range(len(indices)) if i < len(indices) - 1 and indices[i] < indices[i+1])
        max_order = len(indices) - 1 if indices else 1
        score += 10.0 * (partial_order / max_order)

    result["score"] = round(min(score, 100.0), 2)
    return result


# ======================================================================
# Running ollama
# ======================================================================

def run_ollama(model: str, prompt: str, timeout: int) -> tuple:
    """Returns (stdout, stderr, returncode, timed_out)."""
    cmd = ["ollama", "run", model, prompt, "--verbose"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "ignore")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "ignore")
        return stdout, stderr, -1, True


def discover_installed_models() -> List[str]:
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
    except Exception:
        return []
    lines = proc.stdout.strip().splitlines()
    models = []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def warmup_model(model: str, timeout: int = 180) -> None:
    """Loads the model into memory so the first timed run isn't skewed by load time."""
    try:
        subprocess.run(
            ["ollama", "run", model, "Reply with just the word OK."],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception:
        pass


# ======================================================================
# Benchmark run + result data structures
# ======================================================================

@dataclass
class RunResult:
    model: str
    task_id: str
    category: str
    eval_type: str
    iteration: int
    success: bool
    wall_time_s: float
    stdout: str
    stderr: str
    stats: Dict[str, Any] = field(default_factory=dict)
    power: Dict[str, Any] = field(default_factory=dict)
    eval_result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def run_benchmark(model: str, task: Task, timeout: int, power: bool,
                   gpu_index: str, power_interval: float, iteration: int) -> RunResult:
    monitor = None
    power_summary: Dict[str, Any] = {}
    if power:
        monitor = PowerMonitor(gpu_index=gpu_index, interval=power_interval)
        monitor.start()

    start = time.time()
    stdout, stderr, returncode, timed_out = run_ollama(model, task.prompt, timeout)
    wall_time = time.time() - start

    if monitor is not None:
        monitor.stop()
        power_summary = monitor.summary()

    stats = parse_verbose_stats(stderr)
    success = (returncode == 0) and not timed_out
    error = None
    if timed_out:
        error = f"Timed out after {timeout}s"
    elif returncode != 0:
        error = f"ollama exited with code {returncode}: {stderr[:300]}"

    eval_result: Dict[str, Any] = {}
    if success:
        if task.eval_type == "python_exec":
            eval_result = evaluate_python_exec(stdout, task.test_code or "", timeout=task.timeout)
        elif task.eval_type == "tool_json":
            eval_result = evaluate_tool_json(stdout, task.required_tools)

    return RunResult(
        model=model, task_id=task.id, category=task.category, eval_type=task.eval_type,
        iteration=iteration, success=success, wall_time_s=wall_time,
        stdout=stdout, stderr=stderr, stats=stats, power=power_summary,
        eval_result=eval_result, error=error,
    )


# ======================================================================
# Aggregation & composite scoring
# ======================================================================

def aggregate_model(results: List[RunResult]) -> Dict[str, Any]:
    correctness = [r.eval_result.get("score", 0.0) for r in results
                   if r.eval_type == "python_exec" and r.success]
    tool_use = [r.eval_result.get("score", 0.0) for r in results
                if r.eval_type == "tool_json" and r.success]
    eval_rates = [r.stats["eval_rate"] for r in results if r.stats.get("eval_rate")]
    avg_powers = [r.power["avg_power_w"] for r in results if r.power.get("avg_power_w")]

    tokens_per_joule = []
    for r in results:
        count = r.stats.get("eval_count")
        energy = r.power.get("energy_j")
        if count and energy:
            tokens_per_joule.append(count / energy)

    success_rate = (sum(1 for r in results if r.success) / len(results)) if results else 0.0

    return {
        "runs": len(results),
        "success_rate": round(success_rate * 100, 1),
        "avg_correctness": round(statistics.mean(correctness), 2) if correctness else 0.0,
        "avg_tool_use": round(statistics.mean(tool_use), 2) if tool_use else 0.0,
        "avg_eval_rate_tps": round(statistics.mean(eval_rates), 2) if eval_rates else 0.0,
        "avg_power_w": round(statistics.mean(avg_powers), 2) if avg_powers else 0.0,
        "avg_tokens_per_joule": round(statistics.mean(tokens_per_joule), 3) if tokens_per_joule else 0.0,
    }


def normalize_relative(value: float, max_value: float) -> float:
    if not max_value:
        return 0.0
    return max(0.0, min(100.0, 100.0 * value / max_value))


def compute_composite_scores(aggregates: Dict[str, Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    max_rate = max((a["avg_eval_rate_tps"] for a in aggregates.values()), default=0.0)
    max_tpj = max((a["avg_tokens_per_joule"] for a in aggregates.values()), default=0.0)

    composite = {}
    for model, a in aggregates.items():
        speed_norm = normalize_relative(a["avg_eval_rate_tps"], max_rate)
        power_norm = normalize_relative(a["avg_tokens_per_joule"], max_tpj)
        score = (
            weights["correctness"] * a["avg_correctness"]
            + weights["tool_use"] * a["avg_tool_use"]
            + weights["speed"] * speed_norm
            + weights["power"] * power_norm
        )
        composite[model] = {
            **a,
            "speed_score_norm": round(speed_norm, 2),
            "power_efficiency_score_norm": round(power_norm, 2),
            "composite_score": round(score, 2),
        }
    return composite


# ======================================================================
# Reporting
# ======================================================================

def build_leaderboard(composite: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Returns the composite results ranked best-to-worst, with an explicit rank field."""
    ranked = sorted(composite.items(), key=lambda kv: kv[1]["composite_score"], reverse=True)
    return [{"rank": i, "model": model, **a} for i, (model, a) in enumerate(ranked, start=1)]


def print_leaderboard(composite: Dict[str, Dict[str, Any]], weights: Dict[str, float]) -> None:
    leaderboard = build_leaderboard(composite)

    headers = ["Rank", "Model", "Composite", "Correctness", "Tool-Use", "Success%",
               "Tok/s", "Avg W", "Tok/J"]
    rows = []
    for entry in leaderboard:
        rows.append([
            str(entry["rank"]), entry["model"], f"{entry['composite_score']:.1f}",
            f"{entry['avg_correctness']:.1f}", f"{entry['avg_tool_use']:.1f}",
            f"{entry['success_rate']:.0f}", f"{entry['avg_eval_rate_tps']:.1f}",
            f"{entry['avg_power_w']:.1f}", f"{entry['avg_tokens_per_joule']:.3f}",
        ])

    widths = [max(len(headers[c]), max((len(r[c]) for r in rows), default=0)) for c in range(len(headers))]

    def fmt_row(row):
        return "  ".join(cell.ljust(widths[c]) for c, cell in enumerate(row))

    print()
    print("=" * 100)
    print("BENCHMARK LEADERBOARD")
    print(f"(weights: correctness={weights['correctness']}, tool_use={weights['tool_use']}, "
          f"speed={weights['speed']}, power_efficiency={weights['power']})")
    print("=" * 100)
    print(fmt_row(headers))
    print("-" * 100)
    for row in rows:
        print(fmt_row(row))
    print("=" * 100)

    if leaderboard:
        top_a = leaderboard[0]
        print(f"\nRecommended model: {top_a['model']}")
        print(f"  - Composite score: {top_a['composite_score']:.1f}/100")
        print(f"  - Code correctness: {top_a['avg_correctness']:.1f}/100 "
              f"(success rate {top_a['success_rate']:.0f}%)")
        print(f"  - Tool-use / agentic planning score: {top_a['avg_tool_use']:.1f}/100")
        print(f"  - Generation speed: {top_a['avg_eval_rate_tps']:.1f} tokens/s")
        if top_a["avg_power_w"]:
            print(f"  - Avg GPU power draw: {top_a['avg_power_w']:.1f} W "
                  f"({top_a['avg_tokens_per_joule']:.3f} tokens/J)")
    print()


def save_reports(all_results: List[RunResult], composite: Dict[str, Dict[str, Any]], output_dir: Path,
                  weights: Optional[Dict[str, float]] = None,
                  system_info: Optional[Dict[str, Any]] = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    leaderboard = build_leaderboard(composite)

    json_path = output_dir / f"benchmark_{timestamp}.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "system_info": system_info or {},
        "weights": weights or {},
        "runs": [asdict(r) for r in all_results],
        "aggregates": composite,
        "leaderboard": leaderboard,
    }, indent=2))

    return json_path


# ======================================================================
# CLI / main
# ======================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark Ollama models on coding, tool-use, and agentic-planning "
                    "tasks, tracking code correctness, generation speed, and GPU power usage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=None,
                    help="Ollama model names to benchmark (e.g. qwen2.5-coder:7b). "
                         "Defaults to every model reported by `ollama list`.")
    p.add_argument("--tasks", nargs="+", default=None, choices=[t.id for t in TASKS],
                    help="Subset of task IDs to run. Defaults to all built-in tasks.")
    p.add_argument("--list-tasks", action="store_true", help="Print the built-in tasks and exit.")
    p.add_argument("--iterations", type=int, default=1,
                    help="Number of times to repeat each task per model (default: 1).")
    p.add_argument("--timeout", type=int, default=300,
                    help="Per-run timeout in seconds (default: 300).")
    p.add_argument("--output-dir", type=str, default="benchmark_results",
                    help="Directory to write JSON reports to.")
    p.add_argument("--power", dest="power", action="store_true", default=True,
                    help="Monitor GPU power via nvidia-smi (default: enabled).")
    p.add_argument("--no-power", dest="power", action="store_false",
                    help="Disable GPU power monitoring.")
    p.add_argument("--gpu-index", type=str, default="0",
                    help="GPU index passed to nvidia-smi, or 'all' to sum across all GPUs.")
    p.add_argument("--power-interval", type=float, default=0.5,
                    help="Seconds between power samples (default: 0.5).")
    p.add_argument("--no-warmup", dest="warmup", action="store_false", default=True,
                    help="Skip the warmup run that loads each model into memory first.")
    p.add_argument("--weight-correctness", type=float, default=0.4)
    p.add_argument("--weight-tool-use", type=float, default=0.2)
    p.add_argument("--weight-speed", type=float, default=0.2)
    p.add_argument("--weight-power", type=float, default=0.2)
    return p


def preflight_checks(power_requested: bool) -> bool:
    ok = True
    if shutil.which("ollama") is None:
        print("ERROR: `ollama` executable not found on PATH. Install Ollama first.", file=sys.stderr)
        ok = False
    else:
        try:
            subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10, check=True)
        except Exception as e:
            print(f"ERROR: `ollama list` failed - is `ollama serve` running? ({e})", file=sys.stderr)
            ok = False

    if power_requested and not nvidia_smi_available():
        print("WARNING: `nvidia-smi` not found. Continuing without power monitoring.", file=sys.stderr)

    return ok


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.list_tasks:
        for t in TASKS:
            print(f"- {t.id} [{t.category}/{t.eval_type}]: {t.prompt.splitlines()[0][:90]}...")
        return 0

    if not preflight_checks(power_requested=args.power):
        return 1

    power_enabled = args.power and nvidia_smi_available()

    # Get installed models for potential interactive selection
    installed_models = discover_installed_models()
    
    if args.models:
        models = args.models
    elif installed_models:
        # Use interactive model selection when no --models specified
        models = prompt_model_selection(installed_models)
        if not models:
            print("ERROR: no models selected.", file=sys.stderr)
            return 1
    else:
        print("ERROR: no models specified and none discovered via `ollama list`.", file=sys.stderr)
        return 1

    tasks = [TASKS_BY_ID[t] for t in args.tasks] if args.tasks else TASKS

    weights = {
        "correctness": args.weight_correctness,
        "tool_use": args.weight_tool_use,
        "speed": args.weight_speed,
        "power": args.weight_power,
    }

    print(f"Models: {', '.join(models)}")
    print(f"Tasks:  {', '.join(t.id for t in tasks)}")
    print(f"Iterations per task: {args.iterations} | Power monitoring: {power_enabled}\n")

    system_info = get_system_info()
    cpu = system_info["cpu"]
    mem = system_info["memory"]
    gpu_names = ", ".join(g["name"] for g in system_info["gpus"]) or "none detected"
    print("System info:")
    print(f"  OS:      {system_info['os']}")
    print(f"  CPU:     {cpu['model']} ({cpu['logical_cores']} logical cores, {cpu['architecture']})")
    print(f"  RAM:     {mem['total_gb']} GB" if mem.get("total_gb") else "  RAM:     unknown")
    print(f"  GPU(s):  {gpu_names}")
    print(f"  Ollama:  {system_info['ollama_version'] or 'unknown'}\n")

    all_results: List[RunResult] = []

    for model in models:
        if args.warmup:
            print(f"[{model}] warming up...")
            warmup_model(model)

        model_results: List[RunResult] = []
        for task in tasks:
            for it in range(1, args.iterations + 1):
                label = f"[{model}] task={task.id} iter={it}/{args.iterations}"
                print(f"{label} running...", end=" ", flush=True)
                result = run_benchmark(
                    model=model, task=task, timeout=args.timeout, power=power_enabled,
                    gpu_index=args.gpu_index, power_interval=args.power_interval, iteration=it,
                )
                model_results.append(result)
                all_results.append(result)

                status = "OK" if result.success else f"FAILED ({result.error})"
                rate = result.stats.get("eval_rate")
                watt = result.power.get("avg_power_w")
                extra = []
                if rate:
                    extra.append(f"{rate:.1f} tok/s")
                if watt:
                    extra.append(f"{watt:.1f} W")
                if result.eval_result:
                    extra.append(f"score={result.eval_result.get('score', 0):.1f}")
                print(f"{status} " + (f"[{', '.join(extra)}]" if extra else ""))

    aggregates = {model: aggregate_model([r for r in all_results if r.model == model]) for model in models}
    composite = compute_composite_scores(aggregates, weights)

    print_leaderboard(composite, weights)

    json_path = save_reports(all_results, composite, Path(args.output_dir),
                              weights=weights, system_info=system_info)
    print(f"Full results:    {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
