#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Ollama Coding Model Benchmark
==============================

Benchmark Ollama models on coding tasks across four dimensions:

1. **Code Correctness**  - the model writes code; we run it against a test suite.
2. **Agentic / Tool Use** - either a one-shot "plan the tools" proxy OR (preferred)
                            a *real* multi-turn agent loop driven through Ollama's
                            native tool-calling API against an in-memory virtual FS.
3. **Generation Speed**   - tokens/second (eval rate).
4. **Power Efficiency**   - tokens/joule, using `nvidia-smi` when available.

Design goals
------------
- **Single file, stdlib only.** No third-party dependencies. Talks to Ollama
  through its HTTP API by default, with an `ollama run` CLI fallback.
- **Reproducible** sampling: temperature, seed, context length, max tokens, top-k/p
  are all controllable and recorded in the report.
- **Robust**: per-evaluation error isolation, checkpointed/incremental result
  saving, Ctrl-C (SIGINT) partial-save, and `--resume`.
- **Rich reporting**: JSON + CSV + Markdown, variance/uncertainty, TTFT,
  optional cost estimate, and cross-run comparison.

Usage
-----
    # quick, single model, all built-in tasks
    python3 ollama_coding_benchmark.py --models llama3.1:8b --iterations 2

    # list the built-in tasks
    python3 ollama_coding_benchmark.py --list-tasks

    # real (native) tool-calling agent loop + absolute-normalized composite
    python3 ollama_coding_benchmark.py --models llama3.1:8b \\
        --tool-mode native --normalize absolute --ref-rate 100 --ref-tpj 1.0

    # compare two previous runs
    python3 ollama_coding_benchmark.py --compare report_a.json report_b.json

See `--help` for the full option list.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

__version__ = "2.0.0"

# ----------------------------------------------------------------------------
# Global stop flag (set by the SIGINT handler) so we can gracefully stop and
# save partial results instead of losing everything.
# ----------------------------------------------------------------------------
_STOP = threading.Event()
_results_lock = threading.Lock()


def _request_stop(signum: int, _frame: Any) -> None:
    """Signal handler: request a graceful stop (main loop checks this flag)."""
    _STOP.set()
    print("\n[!] Stop requested - finishing current step and saving partial results...", file=sys.stderr)




# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class Task:
    """A single benchmark task.

    Attributes:
        id: stable identifier (e.g. "py_fizzbuzz").
        category: human-facing group ("Python", "JavaScript", "Agentic"...).
        eval_type: "code" (generate + run) or "tool" (agentic / tool use).
        language: "python" | "javascript" | "bash" (only for eval_type=="code").
        prompt: the instruction given to the model.
        test_code: verifier. For python it runs after `from solution import *` and
            may use `assert`; for other languages it runs after the solution and
            must exit non-zero on failure.
        required_tools: ordered list of tool names a plan/agent should use (tool tasks).
        timeout: per-evaluation timeout (seconds).
        difficulty: relative weight used when aggregating scores (default 1.0).
        native_files: seed files for the in-memory virtual FS (native tool mode).
        native_prompt: optional prompt override used in native tool mode.
        native_verify: python snippet (given `fs`) used to grade native tool tasks.
    """
    id: str
    category: str
    eval_type: str
    prompt: str
    test_code: str = ""
    required_tools: List[str] = field(default_factory=list)
    timeout: int = 120
    language: str = "python"
    difficulty: float = 1.0
    native_files: Dict[str, str] = field(default_factory=dict)
    native_prompt: str = ""
    native_verify: str = ""


@dataclass
class GenerationStats:
    """Normalized generation statistics (durations in seconds)."""
    prompt_count: int = 0
    prompt_eval_rate_tps: float = 0.0
    eval_count: int = 0
    eval_rate_tps: float = 0.0
    total_duration_s: float = 0.0
    load_duration_s: float = 0.0
    ttft_s: float = 0.0


@dataclass
class RunResult:
    model: str
    task_id: str
    task_category: str
    eval_type: str
    language: str
    difficulty: float
    response_text: str
    success: bool = False
    error: str = ""
    score: float = 0.0
    passed: int = 0
    total: int = 1
    stats: GenerationStats = field(default_factory=GenerationStats)
    power: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: str = ""
    backend: str = ""
    completed: bool = True




def _ns_to_s(value: Any, default: float = 0.0) -> float:
    """Ollama reports durations in nanoseconds; convert to seconds safely."""
    try:
        return float(value) / 1_000_000_000.0
    except (TypeError, ValueError):
        return default


def stats_from_api(payload: Dict[str, Any]) -> GenerationStats:
    """Build GenerationStats from an Ollama /api/generate (or /api/chat) response."""
    def _dur(key: str) -> float:
        return _ns_to_s(payload.get(key))

    prompt_count = int(payload.get("prompt_eval_count", 0) or 0)
    eval_count = int(payload.get("eval_count", 0) or 0)
    prompt_eval_duration = _dur("prompt_eval_duration")
    eval_duration = _dur("eval_duration")
    prompt_rate = (prompt_count / prompt_eval_duration) if prompt_eval_duration > 0 else 0.0
    eval_rate = (eval_count / eval_duration) if eval_duration > 0 else 0.0
    load_duration = _dur("load_duration")
    return GenerationStats(
        prompt_count=prompt_count,
        prompt_eval_rate_tps=round(prompt_rate, 3),
        eval_count=eval_count,
        eval_rate_tps=round(eval_rate, 3),
        total_duration_s=round(_dur("total_duration"), 6),
        load_duration_s=round(load_duration, 6),
        ttft_s=round(load_duration + prompt_eval_duration, 6),
    )


def stats_from_verbose(stderr_text: str) -> GenerationStats:
    """Build GenerationStats by parsing `ollama run --verbose` stderr output."""
    stats = GenerationStats()
    txt = stderr_text or ""
    m = re.search(r"eval rate \(output\):\s*([0-9.]+)\s*tokens/s", txt)
    if m:
        stats.eval_rate_tps = float(m.group(1))
    m = re.search(r"eval rate \(prompt\):\s*([0-9.]+)\s*tokens/s", txt)
    if m:
        stats.prompt_eval_rate_tps = float(m.group(1))
    m = re.search(r"total duration:\s*([0-9.]+)\s*s", txt)
    if m:
        stats.total_duration_s = float(m.group(1))
    m = re.search(r"load duration:\s*([0-9.]+)\s*s", txt)
    if m:
        stats.load_duration_s = float(m.group(1))
    m = re.search(r"prompt eval count:\s*([0-9]+)", txt)
    if m:
        stats.prompt_count = int(m.group(1))
    m = re.search(r"eval count:\s*([0-9]+)", txt)
    if m:
        stats.eval_count = int(m.group(1))
    # Fallback rate derivation if only duration+count are present.
    if stats.eval_rate_tps == 0.0:
        mm = re.search(r"eval duration:\s*([0-9.]+)\s*s", txt)
        if mm and stats.eval_count > 0:
            stats.eval_rate_tps = round(stats.eval_count / float(mm.group(1)), 3)
    # TTFT approximated from load + prompt-eval durations when available.
    pm = re.search(r"prompt eval duration:\s*([0-9.]+)\s*s", txt)
    if stats.ttft_s == 0.0 and (stats.load_duration_s > 0 or pm):
        stats.ttft_s = round(stats.load_duration_s + (float(pm.group(1)) if pm else 0.0), 6)
    return stats




# ----------------------------------------------------------------------------
# Power monitoring (nvidia-smi)
# ----------------------------------------------------------------------------
def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


class PowerMonitor:
    """Continuously sample GPU power draw via `nvidia-smi` in a background thread."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.enabled = nvidia_smi_available()

    def _sample_gpu_power(self) -> Optional[float]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=2,
            ).decode().strip()
            # If multiple GPUs, sum their power draw.
            total = sum(float(x) for x in out.splitlines() if x.strip())
            return total
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            val = self._sample_gpu_power()
            if val is not None:
                self.samples.append(val)
            time.sleep(self.interval)

    def start(self) -> None:
        if self.enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self.samples:
            avg = sum(self.samples) / len(self.samples)
            return {
                "gpu_power_avg_w": round(avg, 2),
                "gpu_power_min_w": round(min(self.samples), 2),
                "gpu_power_max_w": round(max(self.samples), 2),
                "samples": len(self.samples),
            }
        return {"available": False}




# ----------------------------------------------------------------------------
# System / model metadata
# ----------------------------------------------------------------------------
def get_cpu_info() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def get_memory_info() -> str:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return f"{kb / 1024 / 1024:.1f} GB"
    except Exception:
        pass
    return "Unknown"


def get_gpu_info() -> List[str]:
    gpus: List[str] = []
    if not nvidia_smi_available():
        return gpus
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                gpus.append(f"{parts[0]} ({parts[1]})")
    except Exception:
        pass
    return gpus or ["Unknown"]


def get_ollama_version() -> str:
    try:
        out = subprocess.check_output(["ollama", "--version"], stderr=subprocess.DEVNULL,
                                     timeout=5).decode()
        m = re.search(r"version is ([0-9.]+)", out)
        return m.group(1) if m else out.strip().splitlines()[-1]
    except Exception:
        return "Unknown"


def get_model_metadata(client: Optional["OllamaClient"], model: str) -> Dict[str, Any]:
    """Best-effort capture of model size/details from the Ollama API."""
    meta: Dict[str, Any] = {}
    if client is not None:
        try:
            payload = client._get(f"/api/show?name={urllib.request.quote(model)}")
            details = payload.get("details", {})
            if details.get("parameter_size"):
                meta["parameter_size"] = details["parameter_size"]
            if details.get("quantization_level"):
                meta["quantization"] = details["quantization_level"]
            if details.get("family"):
                meta["family"] = details["family"]
        except Exception:
            pass
    if not meta:
        # Fall back to `ollama list` to at least capture the on-disk size.
        try:
            out = subprocess.check_output(["ollama", "list"], stderr=subprocess.DEVNULL,
                                         timeout=5).decode()
            for line in out.splitlines():
                cols = line.split()
                if cols and cols[0] == model and len(cols) >= 3:
                    meta["disk_size"] = cols[2]
                    break
        except Exception:
            pass
    return meta


def get_system_info() -> Dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpus": get_gpu_info(),
        "ollama_version": get_ollama_version(),
        "python": platform.python_version(),
    }




# ----------------------------------------------------------------------------
# Generation options (shared by both backends)
# ----------------------------------------------------------------------------
def build_generation_options(
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the Ollama `options` payload from the CLI parameters (None => default)."""
    opts: Dict[str, Any] = {}
    if temperature is not None:
        opts["temperature"] = temperature
    if seed is not None:
        opts["seed"] = seed
    if num_ctx is not None:
        opts["num_ctx"] = num_ctx
    if num_predict is not None:
        opts["num_predict"] = num_predict
    if top_k is not None:
        opts["top_k"] = top_k
    if top_p is not None:
        opts["top_p"] = top_p
    return opts


class OllamaError(RuntimeError):
    """Raised when the Ollama API returns an error or is unreachable."""


def _default_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


class OllamaClient:
    """Minimal stdlib HTTP client for the Ollama server API."""

    def __init__(self, host: Optional[str] = None, timeout: float = 300.0):
        self.host = (host or _default_host()).rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
                 timeout: Optional[float] = None) -> Dict[str, Any]:
        url = self.host + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            raise OllamaError(f"HTTP {e.code} from Ollama: {detail}") from e
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            raise OllamaError(f"Cannot reach Ollama at {url}: {e}") from e

    def _get(self, path: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._request("GET", path, None, timeout)

    def is_available(self) -> bool:
        try:
            self._get("/api/version", timeout=3)
            return True
        except OllamaError:
            return False

    def list_models(self) -> List[str]:
        try:
            payload = self._get("/api/tags", timeout=5)
            return [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
        except OllamaError:
            return []

    def generate(self, model: str, prompt: str, options: Dict[str, Any],
                 timeout: Optional[float] = None) -> Tuple[str, GenerationStats]:
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            body["options"] = options
        payload = self._request("POST", "/api/generate", body, timeout)
        return payload.get("response", ""), stats_from_api(payload)

    def chat(self, model: str, messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None,
             options: Optional[Dict[str, Any]] = None,
             timeout: Optional[float] = None) -> Dict[str, Any]:
        """Call /api/chat and return the FULL payload (message + stats fields).

        The assistant message is at ``payload["message"]``; per-request stats
        (``prompt_eval_count``, ``eval_count``, ``eval_duration``, ...) are
        top-level keys, which the agent loop sums across turns.
        """
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if tools:
            body["tools"] = tools
        if options:
            body["options"] = options
        return self._request("POST", "/api/chat", body, timeout)


# ----------------------------------------------------------------------------
# CLI fallback backend (ollama run)
# ----------------------------------------------------------------------------
def build_ollama_cli_command(model: str, prompt: str) -> List[str]:
    return ["ollama", "run", model, prompt, "--verbose"]


def run_ollama_cli(model: str, prompt: str, timeout: int) -> Tuple[str, GenerationStats, str]:
    """Run a model via the `ollama run` CLI. Returns (text, stats, error)."""
    if shutil.which("ollama") is None:
        return "", GenerationStats(), "ollama CLI not found"
    try:
        result = subprocess.run(
            build_ollama_cli_command(model, prompt),
            capture_output=True, text=True, timeout=timeout,
        )
        stderr = result.stderr or ""
        text = (result.stdout or "").strip()
        # `ollama run` echoes the prompt first; keep just the completion.
        if text.startswith(prompt.strip()):
            text = text[len(prompt.strip()):].strip()
        return text, stats_from_verbose(stderr), result.stderr
    except subprocess.TimeoutExpired as e:
        return (e.stdout or "").strip(), GenerationStats(), f"timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001 - surface any CLI failure as a soft error
        return "", GenerationStats(), str(e)




# ----------------------------------------------------------------------------
# Code extraction + execution/evaluation harness
# ----------------------------------------------------------------------------
def extract_code_blocks(response_text: str, language: str = "python") -> List[str]:
    """Extract fenced code blocks, preferring the requested language."""
    text = response_text or ""
    pattern = r"```(\w*)\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    matched = [b[1].strip() for b in blocks if b[0].strip().lower() == language]
    if matched:
        return matched
    any_block = [b[1].strip() for b in blocks if b[1].strip()]
    if any_block:
        return any_block
    return [text.strip()] if text.strip() else []


def _isolated_env() -> Dict[str, str]:
    """A minimal, scrubbed environment for running untrusted model-generated code.

    Note: this is *not* a security sandbox (no seccomp/network-egress blocking);
    it only limits the environment and working directory. Run untrusted output at
    your own risk.
    """
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONIOENCODING": "utf-8",
        "LANG": "C.UTF-8",
        "HOME": os.environ.get("HOME", ""),
    }


def _run_python_granular(task: Task, code: str) -> Tuple[bool, float, int, int, str]:
    """Run a python solution + its verifier. Returns (success, score, passed, total, error)."""
    workdir = tempfile.mkdtemp(prefix="obm_")
    try:
        with open(os.path.join(workdir, "solution.py"), "w") as f:
            f.write(code)
        # Checks are authored as `lambda: assert ...`; `assert` is a statement,
        # so normalize it into a boolean-returning lambda before exec.
        test_code = re.sub(r"lambda:\s*assert\s+", "lambda: ", task.test_code)
        test_src = (
            "import sys\n"
            f"sys.path.insert(0, r'{workdir}')\n"
            "passed = 0\n"
            "total = 0\n"
            "errors = []\n"
            "from solution import *  # noqa: F401,F403\n"
            "\n"
            "def __run(test_name, fn):\n"
            "    global passed, total\n"
            "    total += 1\n"
            "    try:\n"
            "        result = fn()\n"
            "        if result is not None and not result:\n"
            "            raise AssertionError(f'{test_name} is falsy')\n"
            "        passed += 1\n"
            "    except Exception as exc:  # noqa: BLE001\n"
            "        errors.append(f'{test_name}: {exc}')\n"
            "\n"
            + test_code
            + "\n"
            "import json as _json\n"
            "print(_json.dumps({'passed': passed, 'total': total, 'errors': errors}))\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", test_src],
                capture_output=True, text=True,
                timeout=task.timeout, cwd=workdir, env=_isolated_env(),
            )
        except subprocess.TimeoutExpired:
            return False, 0.0, 0, 1, f"test timed out after {task.timeout}s"
        if proc.returncode != 0:
            return False, 0.0, 0, 1, (proc.stderr or "non-zero exit").strip()[:500]
        last_line = (proc.stdout or "").strip().splitlines()[-1:]
        try:
            info = json.loads(last_line[0])
            passed, total = int(info.get("passed", 0)), int(info.get("total", 1))
            total = max(total, 1)
            score = round(100.0 * passed / total, 2)
            success = passed == total
            error = "; ".join(info.get("errors", [])[:5])
            return success, score, passed, total, error
        except (ValueError, IndexError, KeyError):
            return False, 0.0, 0, 1, (proc.stderr or "could not parse test result").strip()[:500]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def evaluate_python_exec(task: Task, response_text: str) -> Dict[str, Any]:
    """Evaluate a python code task. Returns a dict of evaluation fields."""
    blocks = extract_code_blocks(response_text, "python")
    code = blocks[0] if blocks else ""
    if not code or "def " not in code:
        return {"success": False, "error": "No Python function found in response",
                "score": 0.0, "passed": 0, "total": 1}
    success, score, passed, total, error = _run_python_granular(task, code)
    return {"success": success, "error": error, "score": score,
            "passed": passed, "total": total}


def evaluate_code_response(task: Task, response_text: str) -> Dict[str, Any]:
    """Dispatch code evaluation to the right language harness."""
    if task.eval_type != "code":
        raise ValueError(f"evaluate_code_response called on non-code task {task.id}")
    if task.language == "python":
        return evaluate_python_exec(task, response_text)
    if task.language == "javascript":
        return _run_language_task(task, response_text, "javascript", "node", ".js")
    if task.language == "bash":
        return _run_language_task(task, response_text, "bash", "bash", ".sh")
    return {"success": False, "error": f"unsupported language: {task.language}",
            "score": 0.0, "passed": 0, "total": 1}




def _run_language_task(
    task: Task, response_text: str, language: str, interpreter: str, ext: str,
) -> Dict[str, Any]:
    """Run a non-python code task: concatenate solution + test and check exit code."""
    if shutil.which(interpreter) is None:
        return {"success": False, "error": f"'{interpreter}' not found (task skipped)",
                "score": 0.0, "passed": 0, "total": 1, "skipped": True}
    blocks = extract_code_blocks(response_text, language)
    code = blocks[0] if blocks else ""
    if not code:
        return {"success": False, "error": "No code block found in response",
                "score": 0.0, "passed": 0, "total": 1}
    workdir = tempfile.mkdtemp(prefix="obm_")
    try:
        sol_path = os.path.join(workdir, "solution" + ext)
        with open(sol_path, "w") as f:
            f.write(code)
        # JavaScript: surface the declared function as a global so the test can
        # call it directly (CommonJS `require` only returns `module.exports`).
        if language == "javascript":
            mfn = re.search(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", code)
            if not mfn:
                mfn = re.search(r"\bconst\s+([A-Za-z_$][\w$]*)\s*=\s*(function|\()", code)
            if mfn:
                with open(sol_path, "a") as f:
                    f.write("\nif (typeof globalThis !== 'undefined') "
                            f"{{ globalThis.{mfn.group(1)} = {mfn.group(1)}; }}\n")
        with open(os.path.join(workdir, "test" + ext), "w") as f:
            # test_code references the solution; run both in the same interpreter.
            if language == "javascript":
                f.write("const solution = (typeof require !== 'undefined') ? require('./solution.js') : {};\n")
                f.write(task.test_code)
            else:  # bash
                f.write(f". ./solution{ext}\n")
                f.write(task.test_code)
        runner = interpreter if language == "bash" else interpreter
        entry = "test" + ext
        try:
            proc = subprocess.run(
                [runner, entry],
                capture_output=True, text=True, timeout=task.timeout,
                cwd=workdir, env=_isolated_env(),
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"timed out after {task.timeout}s",
                    "score": 0.0, "passed": 0, "total": 1}
        if proc.returncode == 0:
            return {"success": True, "error": "", "score": 100.0, "passed": 1, "total": 1}
        return {"success": False, "error": (proc.stderr or "test failed").strip()[:500],
                "score": 0.0, "passed": 0, "total": 1}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def evaluate_tool_json(task: Task, response_text: str) -> Dict[str, Any]:
    """Evaluate the tool-use *proxy*: a single-shot JSON plan of tools to call.

    Scoring (0-100):
        - parseable JSON array: +10
        - each required tool present: +20 each (up to 100)
        - exact ordered match: +50 bonus (capped at 100)
        - a valid JSON object with a list field: small structure bonus

    A partial match (some but not all tools) is fully supported and scores
    proportionally. (Fixed: earlier version referenced an undefined `indices`
    in this branch, raising `UnboundLocalError`.)
    """
    text = (response_text or "").strip()
    json_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"success": False, "error": "Invalid JSON format", "score": 0.0,
                "passed": 0, "total": 1}

    tools_used: List[str] = []
    if isinstance(data, list):
        tools_used = [str(x).strip() for x in data if isinstance(x, (str, int, float))]
    elif isinstance(data, dict):
        for key in ("tools", "actions", "calls", "steps", "functions"):
            v = data.get(key)
            if isinstance(v, list):
                tools_used = [str(x).strip() for x in v if isinstance(x, (str, int, float))]
                break

    required = list(task.required_tools)
    if not required:
        return {"success": isinstance(data, (list, dict)) and bool(tools_used),
                "error": "" if tools_used else "No tools found",
                "score": 100.0 if tools_used else 0.0,
                "passed": 1 if tools_used else 0, "total": 1}

    found = [t for t in required if t in tools_used]
    score = min(len(found) * (100.0 / len(required)), 100.0)
    error = ""

    if required and all(t in tools_used for t in required):
        indices = [tools_used.index(t) for t in required]
        if indices == sorted(indices):
            score = 100.0
        else:
            score = min(score + 50.0, 100.0)
    elif found:
        missing = [t for t in required if t not in tools_used]
        error = f"missing tools: {', '.join(missing)}"

    success = score >= 80.0
    return {"success": success, "error": error, "score": round(score, 2),
            "passed": len(found), "total": len(required)}




# ----------------------------------------------------------------------------
# Task set (v2.0: expanded, harder, multi-language, tool-proxy + native-agent)
# ----------------------------------------------------------------------------
def difficulty_label(weight: float) -> str:
    """Map a numeric difficulty weight to a human label."""
    if weight <= 1.0:
        return "easy"
    if weight <= 1.5:
        return "medium"
    return "hard"


TASKS: List[Task] = [
    # ----------------------------- Python: easy -----------------------------
    Task(
        id="py_reverse_string", category="Python", eval_type="code", language="python",
        difficulty=1.0,
        prompt="Write a Python function `reverse_string(s: str) -> str` that returns the "
               "reversed string. Return ONLY the code in a single ```python block.",
        test_code=(
            "__run('abc', lambda: assert reverse_string('abc') == 'cba')\n"
            "__run('empty', lambda: assert reverse_string('') == '')\n"
            "__run('palin', lambda: assert reverse_string('aba') == 'aba')\n"
            "__run('spaces', lambda: assert reverse_string('a b') == 'b a')\n"
        ),
    ),
    Task(
        id="py_fizzbuzz", category="Python", eval_type="code", language="python",
        difficulty=1.0,
        prompt="Write a Python function `fizzbuzz(n: int) -> list` that returns a list of the "
               "FizzBuzz strings for 1..n inclusive (12 -> 'FizzBuzz', 15 -> 'FizzBuzz'). "
               "Return ONLY the code in a single ```python block.",
        test_code=(
            "__run('1-3', lambda: fizzbuzz(3) == ['1', '2', 'Fizz'])\n"
            "__run('1-10', lambda: fizzbuzz(10) == ['1', '2', 'Fizz', '4', 'Buzz', 'Fizz', '7', '8', 'Fizz', 'Buzz'])\n"
            "__run('12-15', lambda: fizzbuzz(15)[11:] == ['Fizz', '13', '14', 'FizzBuzz'])\n"
        ),
    ),
    Task(
        id="py_is_palindrome", category="Python", eval_type="code", language="python",
        difficulty=1.0,
        prompt="Write a Python function `is_palindrome(s: str) -> bool` that ignores "
               "case, spaces and punctuation and returns True if the result is a palindrome. "
               "Return ONLY the code in a single ```python block.",
        test_code=(
            "__run('racecar', lambda: assert is_palindrome('racecar') is True)\n"
            "__run('clean', lambda: assert is_palindrome('A man, a plan, a canal: Panama') is True)\n"
            "__run('no', lambda: assert is_palindrome('hello') is False)\n"
            "__run('empty', lambda: assert is_palindrome('') is True)\n"
        ),
    ),
    Task(
        id="py_count_vowels", category="Python", eval_type="code", language="python",
        difficulty=1.0,
        prompt="Write a Python function `count_vowels(s: str) -> int` that counts the vowel "
               "characters (a e i o u, case-insensitive). Return ONLY the code in a single ```python block.",
        test_code=(
            "__run('hello', lambda: assert count_vowels('hello') == 2)\n"
            "__run('AEIOU', lambda: assert count_vowels('AEIOU') == 5)\n"
            "__run('none', lambda: assert count_vowels('rhythm') == 0)\n"
            "__run('empty', lambda: assert count_vowels('') == 0)\n"
        ),
    ),
    Task(
        id="py_matrix_transpose", category="Python", eval_type="code", language="python",
        difficulty=1.0,
        prompt="Write a Python function `matrix_transpose(m: list) -> list` that returns the "
               "transpose of a rectangular 2D list. Return ONLY the code in a single ```python block.",
        test_code=(
            "__run('rect', lambda: assert matrix_transpose([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]])\n"
            "__run('square', lambda: assert matrix_transpose([[1,2],[3,4]]) == [[1,3],[2,4]])\n"
            "__run('single', lambda: assert matrix_transpose([[1,2,3]]) == [[1],[2],[3]])\n"
        ),
    ),
    # ----------------------------- Python: medium ---------------------------
    Task(
        id="py_binary_search", category="Python", eval_type="code", language="python",
        difficulty=1.5,
        prompt="Write a Python function `binary_search(nums: list, target) -> int` that returns "
               "the index of target in the sorted list `nums`, or -1 if absent. Use binary search. "
               "Return ONLY the code in a single ```python block.",
        test_code=(
            "__run('found', lambda: assert binary_search([1,3,5,7,9], 7) == 3)\n"
            "__run('first', lambda: assert binary_search([2,4,6], 2) == 0)\n"
            "__run('last', lambda: assert binary_search([2,4,6], 6) == 2)\n"
            "__run('absent', lambda: assert binary_search([1,3,5], 4) == -1)\n"
            "__run('empty', lambda: assert binary_search([], 1) == -1)\n"
        ),
    ),
    Task(
        id="py_two_sum", category="Python", eval_type="code", language="python",
        difficulty=1.5,
        prompt="Write a Python function `two_sum(nums: list, target) -> list` that returns the "
               "indices (sorted ascending) of the two distinct elements whose sum is target. "
               "Return ONLY the code in a single ```python block.",
        test_code=(
            "__run('basic', lambda: assert two_sum([2,7,11,15], 9) == [0,1])\n"
            "__run('other', lambda: assert two_sum([3,2,4], 6) == [1,2])\n"
            "__run('neg', lambda: assert two_sum([-1,-7,-5,3], -6) == [0,2])\n"
        ),
    ),
    Task(
        id="py_word_frequency", category="Python", eval_type="code", language="python",
        difficulty=1.5,
        prompt="Write a Python function `word_frequency(text: str, top_n: int = 0) -> list` that "
               "returns the top_n most common words (case-insensitive) as (word, count) tuples sorted "
               "by count desc then word asc. If top_n is 0, return all. "
               "Return ONLY the code in a single ```python block.",
        test_code=(
            "__run('all', lambda: assert word_frequency('the cat the dog the') == [('the',3),('cat',1),('dog',1)])\n"
            "__run('top1', lambda: assert word_frequency('a a b c', 1) == [('a',2)])\n"
            "__run('case', lambda: assert word_frequency('Apple apple BANANA banana', 2) == [('apple',2),('banana',2)])\n"
        ),
    ),
    Task(
        id="py_merge_intervals", category="Python", eval_type="code", language="python",
        difficulty=1.5,
        prompt="Write a Python function `merge_intervals(intervals: list) -> list` that merges all "
               "overlapping or touching intervals and returns them sorted. Input is a list of "
               "[start, end]. Return ONLY the code in a single ```python block.",
        test_code=(
            "__run('overlap', lambda: assert merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]])\n"
            "__run('touch', lambda: assert merge_intervals([[1,4],[4,5]]) == [[1,5]])\n"
            "__run('contain', lambda: assert merge_intervals([[1,10],[2,5],[3,4]]) == [[1,10]])\n"
            "__run('none', lambda: assert merge_intervals([[1,2],[5,6]]) == [[1,2],[5,6]])\n"
        ),
    ),
    Task(
        id="py_email_extract", category="Python", eval_type="code", language="python",
        difficulty=1.5,
        prompt="Write a Python function `extract_emails(text: str) -> list` that returns all valid "
               "email addresses found in the text, in order of appearance (a simple local@domain "
               "form is sufficient). Return ONLY the code in a single ```python block.",
        test_code=(
            "__run('one', lambda: assert extract_emails('mail me at bob@example.com now') == ['bob@example.com'])\n"
            "__run('two', lambda: assert extract_emails('a@b.com and c@d.io') == ['a@b.com', 'c@d.io'])\n"
            "__run('none', lambda: assert extract_emails('no mail here') == [])\n"
        ),
    ),
    # ------------------------------ Python: hard ----------------------------
    Task(
        id="py_lru_cache", category="Python", eval_type="code", language="python",
        difficulty=2.0,
        prompt="Write a Python class `LRUCache(capacity: int)` with `get(key)` returning the value "
               "or -1, and `put(key, value)` that inserts/updates and evicts the least-recently-used "
               "item when over capacity. Both should be O(1) average. "
               "Return ONLY the code in a single ```python block.",
        test_code=(
            "def __lru(cap, ops):\n"
            "    c = LRUCache(cap)\n"
            "    out = []\n"
            "    for op, a, b in ops:\n"
            "        if op == 'put':\n"
            "            c.put(a, b)\n"
            "        elif op == 'get':\n"
            "            out.append(c.get(a))\n"
            "    return out\n"
            "__run('basic', lambda: __lru(2, [('put',1,1), ('put',2,2), ('get',1,0)]) == [1])\n"
            "__run('evict', lambda: __lru(2, [('put',1,1), ('put',2,2), ('put',3,3), ('get',1,0)]) == [-1])\n"
            "__run('recency', lambda: __lru(2, [('put',1,1), ('put',2,2), ('get',1,0), ('put',3,3), ('get',2,0)]) == [1, -1])\n"
            "__run('update', lambda: __lru(2, [('put',1,1), ('put',1,2), ('get',1,0)]) == [2])\n"
        ),
    ),
    Task(
        id="py_bfs_shortest_path", category="Python", eval_type="code", language="python",
        difficulty=2.0,
        prompt="Write a Python function `shortest_path(graph: dict, start, end) -> int` that returns "
               "the number of edges in the shortest path from start to end in an unweighted graph "
               "({node: [neighbors]}), or -1 if unreachable. Use BFS. "
               "Return ONLY the code in a single ```python block.",
        test_code=(
            "__run('direct', lambda: assert shortest_path({'a':['b'],'b':['a']}, 'a','b') == 1)\n"
            "__run('hop', lambda: assert shortest_path({'a':['b'],'b':['c'],'c':[]}, 'a','c') == 2)\n"
            "__run('none', lambda: assert shortest_path({'a':['b'],'c':['d']}, 'a','d') == -1)\n"
            "__run('same', lambda: assert shortest_path({'a':['b']}, 'a','a') == 0)\n"
        ),
    ),
    Task(
        id="py_token_bucket", category="Python", eval_type="code", language="python",
        difficulty=2.0,
        prompt="Write a Python class `TokenBucket(capacity: int, rate: float)` (rate tokens/second) "
               "with `try_acquire(tokens: int = 1, now: float = 0.0) -> bool`. It starts full, adds "
               "tokens over time according to `now`, caps at capacity, and returns whether the "
               "requested tokens were available (consuming them if so). "
               "Return ONLY the code in a single ```python block.",
        test_code=(
            "def __tb(capacity, rate, calls):\n"
            "    b = TokenBucket(capacity, rate)\n"
            "    return [b.try_acquire(t, now) for (t, now) in calls]\n"
            "__run('start_full', lambda: __tb(5, 1, [(5, 0), (1, 0)]) == [True, False])\n"
            "__run('refill', lambda: __tb(5, 1, [(5, 0), (3, 3)]) == [True, True])\n"
            "__run('cap', lambda: __tb(5, 1, [(5, 0), (10, 100)]) == [True, False])\n"
        ),
    ),
    # ------------------------------- JavaScript -----------------------------
    Task(
        id="js_group_by", category="JavaScript", eval_type="code", language="javascript",
        difficulty=1.5,
        prompt="Write a JavaScript function `groupBy(arr, key)` that groups an array of objects by "
               "the value at `key`, returning an object mapping key-value -> array of objects (in "
               "original order). Return ONLY the code in a single ```javascript block.",
        test_code=(
            "const g = groupBy([{c:'a',n:1},{c:'b',n:2},{c:'a',n:3}], 'c');\n"
            "if (g.a.length !== 2 || g.b.length !== 1) throw new Error('group wrong');\n"
            "if (g.a[0].n !== 1 || g.a[1].n !== 3) throw new Error('order wrong');\n"
        ),
    ),
    Task(
        id="js_string_stats", category="JavaScript", eval_type="code", language="javascript",
        difficulty=1.0,
        prompt="Write a JavaScript function `stringStats(s)` returning an object "
               "{vowels, consonants} counting lowercase vowel and consonant letters (ignore "
               "non-letters). Return ONLY the code in a single ```javascript block.",
        test_code=(
            "const r = stringStats('hello');\n"
            "if (r.vowels !== 2 || r.consonants !== 3) throw new Error('stats wrong: ' + JSON.stringify(r));\n"
            "const r2 = stringStats('AEIOU');\n"
            "if (r2.vowels !== 0 || r2.consonants !== 0) throw new Error('non-lowercase should be ignored');\n"
        ),
    ),
    # --------------------------------- Bash ---------------------------------
    Task(
        id="bash_count_lines", category="Bash", eval_type="code", language="bash",
        difficulty=1.0,
        prompt="Write a bash function `count_lines <file>` that echoes the number of lines in the "
               "file. Return ONLY the code in a single ```bash block.",
        test_code=(
            "printf 'a\\nb\\nc\\n' > in.txt\n"
            "out=$(count_lines in.txt)\n"
            "if [ \"$out\" != \"3\" ]; then echo \"expected 3 got $out\"; exit 1; fi\n"
        ),
    ),
    # --------------------------- Tool-use (JSON plan) ------------------------
    # These are evaluated as a single-shot JSON plan (proxy mode) OR, when a
    # native tool-calling model is used, through the real agent loop.
    Task(
        id="tool_lookup_weather", category="Tool Use", eval_type="tool",
        difficulty=1.0,
        prompt="You have access to these tools: [lookup_weather, lookup_forecast, get_timezone]. "
               "Task: tell the user the current weather in Paris. Respond with ONLY a JSON array "
               "listing the tool names to call, in the correct order. Nothing else.",
        required_tools=["lookup_weather"],
    ),
    Task(
        id="tool_send_email", category="Tool Use", eval_type="tool",
        difficulty=1.5,
        prompt="You have access to these tools: [read_inbox, search_emails, compose_email, send_email, "
               "delete_email]. Task: find the email from 'Alice' about the launch, draft a reply "
               "scheduling it for Friday, and send it. Respond with ONLY a JSON array listing the "
               "tool names to call, in the correct order. Nothing else.",
        required_tools=["search_emails", "compose_email", "send_email"],
    ),
    Task(
        id="tool_data_pipeline", category="Tool Use", eval_type="tool",
        difficulty=1.5,
        prompt="You have access to these tools: [ingest_source, clean_data, transform_data, "
               "validate_schema, store_result, log_event]. Task: load the raw CSV, clean it, "
               "transform to the target schema, validate, and store it. Respond with ONLY a JSON "
               "array listing the tool names to call, in the correct order. Nothing else.",
        required_tools=["ingest_source", "clean_data", "transform_data", "store_result"],
    ),
    Task(
        id="tool_deploy_pipeline", category="Tool Use", eval_type="tool",
        difficulty=2.0,
        prompt="You have access to these tools: [check_status, pull_changes, build, run_tests, "
               "package, deploy, notify_team]. Task: safely ship the release. Respond with ONLY a "
               "JSON array listing the tool names to call, in the correct order. Nothing else.",
        required_tools=["check_status", "build", "run_tests", "package", "deploy"],
    ),
    # ------------------------- Native agent (tool loop) ----------------------
    # These exercise the real /api/chat tool-calling loop against a live
    # workspace. `native_files` seeds the workspace; `native_verify` is an
    # in-process snippet given a `ws` view that must set `success`/`message`.
    Task(
        id="native_create_and_run", category="Agent", eval_type="tool",
        difficulty=1.5,
        prompt="Create a file `greeting.py` that defines `greet(name)` returning "
               "'Hello, {name}!' and a `main()` that prints `greet('world')`. Run it to confirm "
               "it prints 'Hello, world!'.",
        native_prompt="Create a file `greeting.py` that defines `greet(name)` returning "
                      "'Hello, {name}!' and a `main()` that prints `greet('world')`. Run it to "
                      "confirm it prints 'Hello, world!'.",
        native_verify="""
success = False
message = ''
if not ws.exists('greeting.py'):
    message = 'greeting.py was not created'
else:
    rc, out, err = ws.run_python('greeting.py')
    lines = [l for l in (out or '').splitlines() if l.strip()]
    success = (rc == 0 and lines and 'Hello, world!' in lines[-1])
    message = ((out or '') + (err or '')).strip()[:200]
""",
    ),
    Task(
        id="native_fix_buggy", category="Agent", eval_type="tool",
        difficulty=2.0,
        prompt="There is a file `calc.py` whose `add(a, b)` is buggy (returns a - b). Read it, fix "
               "add to return a + b, and run `python calc.py` to confirm it prints 5.",
        native_files={"calc.py": "def add(a, b):\n    return a - b\n\n\ndef main():\n    "
                         "print(add(2, 3))\n\n\nif __name__ == '__main__':\n    main()\n"},
        native_prompt="There is a file `calc.py` whose `add(a, b)` is buggy (returns a - b). Read "
                      "it, fix add to return a + b, and run `python calc.py` to confirm it prints 5.",
        native_verify="""
success = False
message = ''
if not ws.exists('calc.py'):
    message = 'calc.py missing'
else:
    rc, out, err = ws.run_python('calc.py')
    lines = [l for l in (out or '').splitlines() if l.strip()]
    success = (rc == 0 and lines and lines[-1].strip() == '5')
    message = ((out or '') + (err or '')).strip()[:200]
""",
    ),
    Task(
        id="native_multi_file", category="Agent", eval_type="tool",
        difficulty=2.0,
        prompt="Create `utils.py` with `double(x)` returning x*2, and `main.py` that imports "
               "double from utils and prints `double(21)`. Run `main.py` to confirm it prints 42.",
        native_prompt="Create `utils.py` with `double(x)` returning x*2, and `main.py` that imports "
                      "double from utils and prints `double(21)`. Run `main.py` to confirm it "
                      "prints 42.",
        native_verify="""
success = False
message = ''
if not (ws.exists('utils.py') and ws.exists('main.py')):
    message = 'utils.py and/or main.py missing'
else:
    rc, out, err = ws.run_python('main.py')
    lines = [l for l in (out or '').splitlines() if l.strip()]
    success = (rc == 0 and lines and lines[-1].strip() == '42')
    message = ((out or '') + (err or '')).strip()[:200]
""",
    ),
    Task(
        id="native_search_replace", category="Agent", eval_type="tool",
        difficulty=1.5,
        prompt="In `config.txt`, replace the word 'beta' with 'delta' and save the file.",
        native_files={"config.txt": "alpha\nbeta\ngamma\n"},
        native_prompt="In `config.txt`, replace the word 'beta' with 'delta' and save the file.",
        native_verify="""
success = False
message = ''
if not ws.exists('config.txt'):
    message = 'config.txt missing'
else:
    content = ws.read('config.txt')
    success = ('delta' in content and 'beta' not in content)
    message = repr(content)[:200]
""",
    ),
    Task(
        id="native_refactor", category="Agent", eval_type="tool",
        difficulty=2.0,
        prompt="Refactor `legacy.py`: turn the `calc(x)` function (returns x*2) into a method "
               "`double(self, x)` on a class `Doubler`, keeping the file importable. Update the file.",
        native_files={"legacy.py": "def calc(x):\n    return x * 2\n\n\nif __name__ == '__main__':\n    "
                                   "print(calc(21))\n"},
        native_prompt="Refactor `legacy.py`: turn the `calc(x)` function (returns x*2) into a method "
                      "`double(self, x)` on a class `Doubler`, keeping the file importable. Update "
                      "the file.",
        native_verify="""
success = False
message = ''
if not ws.exists('legacy.py'):
    message = 'legacy.py missing'
else:
    import importlib.util
    path = ws.path('legacy.py')
    spec = importlib.util.spec_from_file_location('legacy_mod', path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        d = mod.Doubler()
        val = d.double(21)
        success = (val == 42)
        message = 'Doubler().double(21) = ' + str(val)
    except Exception as e:  # noqa: BLE001
        message = repr(e)[:200]
""",
    ),
]


# ----------------------------------------------------------------------------
# Native agent loop (real /api/chat tool-calling)
# ----------------------------------------------------------------------------
def _guard(root: str, target: str) -> str:
    """Ensure ``target`` stays inside ``root``; return the resolved path."""
    root_real = os.path.realpath(root)
    target_real = os.path.realpath(target)
    if target_real != root_real and not target_real.startswith(root_real + os.sep):
        raise ValueError(f"path escapes workspace: {target}")
    return target_real


def _run_in_ws(cmd: List[str], root: str, timeout: int) -> Tuple[int, str, str]:
    """Run ``cmd`` inside the workspace cwd with an isolated environment."""
    env = _isolated_env()
    env["PWD"] = root
    try:
        p = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", "timeout after %ss" % timeout
    except Exception as e:  # noqa: BLE001
        return 1, "", repr(e)


def _fmt_exec(rc: int, out: str, err: str) -> str:
    parts: List[str] = []
    if out and out.strip():
        parts.append("STDOUT:\n" + out.strip())
    if err and err.strip():
        parts.append("STDERR:\n" + err.strip())
    parts.append("EXIT: %d" % rc)
    return "\n".join(parts)


class WorkspaceView:
    """Read/execute view over a real on-disk workspace directory.

    Exposed to the in-process ``native_verify`` snippets as ``ws``.
    """

    def __init__(self, root: str) -> None:
        self.root = os.path.realpath(root)

    def path(self, rel: str) -> str:
        return _guard(self.root, os.path.abspath(os.path.join(self.root, rel.lstrip("/"))))

    def exists(self, rel: str) -> bool:
        try:
            return os.path.isfile(self.path(rel))
        except Exception:  # noqa: BLE001
            return False

    def read(self, rel: str) -> str:
        try:
            with open(self.path(rel), "r", encoding="utf-8") as f:
                return f.read()
        except Exception:  # noqa: BLE001
            return ""

    def list(self) -> List[str]:
        out: List[str] = []
        for dp, _dirs, files in os.walk(self.root):
            for fn in files:
                full = os.path.join(dp, fn)
                out.append(os.path.relpath(full, self.root))
        return sorted(out)

    def run_python(self, rel: str, timeout: int = 30) -> Tuple[int, str, str]:
        return _run_in_ws([sys.executable, self.path(rel)], self.root, timeout)

    def run_cmd(self, cmdline: str, timeout: int = 30) -> Tuple[int, str, str]:
        return _run_in_ws(["bash", "-c", cmdline], self.root, timeout)


def _tool(name: str, description: str, props: Dict[str, Dict[str, str]], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


NATIVE_TOOLS: List[Dict[str, Any]] = [
    _tool("write_file", "Create or overwrite a file at path with the given content.",
          {"path": {"type": "string", "description": "file path relative to the workspace"},
           "content": {"type": "string", "description": "full file content"}},
          ["path", "content"]),
    _tool("append_file", "Append content to an existing file.",
          {"path": {"type": "string", "description": "file path relative to the workspace"},
           "content": {"type": "string", "description": "content to append"}},
          ["path", "content"]),
    _tool("read_file", "Read and return the contents of a file.",
          {"path": {"type": "string", "description": "file path relative to the workspace"}},
          ["path"]),
    _tool("delete_file", "Delete a file.",
          {"path": {"type": "string", "description": "file path relative to the workspace"}},
          ["path"]),
    _tool("list_files", "List all files in the workspace (relative paths).",
          {}, []),
    _tool("execute_python", "Execute a Python snippet in the workspace and return its output.",
          {"code": {"type": "string", "description": "Python code to run"}},
          ["code"]),
    _tool("shell", "Run a shell command in the workspace and return its output.",
          {"command": {"type": "string", "description": "shell command to run"}},
          ["command"]),
]

def _coerce_args(args: Any) -> Dict[str, Any]:
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return args if isinstance(args, dict) else {}


def _summarize_args(args: Dict[str, Any]) -> str:
    try:
        s = json.dumps(args)
        return s if len(s) <= 60 else s[:57] + "..."
    except (TypeError, ValueError):
        return str(args)[:60]


def execute_tool(name: str, args: Any, root: str, timeout: int = 30) -> str:
    """Execute a single tool call inside the workspace and return a text result."""
    args = _coerce_args(args)

    def target(rel: Any) -> str:
        return _guard(root, os.path.abspath(os.path.join(root, str(rel).lstrip("/"))))

    try:
        if name == "write_file":
            t = target(args["path"])
            os.makedirs(os.path.dirname(t) or ".", exist_ok=True)
            content = str(args.get("content", ""))
            with open(t, "w", encoding="utf-8") as f:
                f.write(content)
            return f"OK wrote {str(args['path'])} ({len(content)} bytes)"
        if name == "append_file":
            t = target(args["path"])
            os.makedirs(os.path.dirname(t) or ".", exist_ok=True)
            content = str(args.get("content", ""))
            with open(t, "a", encoding="utf-8") as f:
                f.write(content)
            return f"OK appended to {str(args['path'])} ({len(content)} bytes)"
        if name == "read_file":
            t = target(args["path"])
            try:
                with open(t, "r", encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                return f"ERROR: file not found: {str(args['path'])}"
        if name == "delete_file":
            t = target(args["path"])
            if os.path.isfile(t):
                os.remove(t)
                return f"OK deleted {str(args['path'])}"
            return f"ERROR: file not found: {str(args['path'])}"
        if name == "list_files":
            out: List[str] = []
            for dp, _dirs, files in os.walk(root):
                for fn in files:
                    out.append(os.path.relpath(os.path.join(dp, fn), root))
            return "\n".join(sorted(out)) or "(workspace is empty)"
        if name == "execute_python":
            code = str(args.get("code", ""))
            tmp = "__agent_exec_" + os.urandom(4).hex() + ".py"
            with open(os.path.join(root, tmp), "w", encoding="utf-8") as f:
                f.write(code)
            rc, out, err = _run_in_ws([sys.executable, os.path.join(root, tmp)], root, timeout)
            try:
                os.remove(os.path.join(root, tmp))
            except OSError:
                pass
            return _fmt_exec(rc, out, err)
        if name == "shell":
            rc, out, err = _run_in_ws(["bash", "-c", str(args.get("command", ""))], root, timeout)
            return _fmt_exec(rc, out, err)
        return f"ERROR: unknown tool '{name}'"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e!r}"


def run_native_agent(
    model: str,
    task: Task,
    client: "OllamaClient",
    options: Dict[str, Any],
    max_turns: int = 6,
    tool_timeout: int = 30,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the real multi-turn tool-calling agent loop for ``task``.

    Returns ``final_text``, ``stats``, ``turns``, ``tool_trace`` and
    ``workspace`` (root path kept alive for verification).
    """
    workspace = tempfile.mkdtemp(prefix="bench_ws_")
    try:
        for rel, content in (task.native_files or {}).items():
            full = _guard(workspace, os.path.abspath(os.path.join(workspace, rel.lstrip("/"))))
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)

        system = (
            "You are a coding agent. Use the provided tools to accomplish the task. "
            "Work step by step, verify your work, and when the task is fully complete "
            "respond with a short final message and do NOT call any tools."
        )
        user_prompt = task.native_prompt or task.prompt
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]

        tot_prompt = tot_eval = tot_dur_ns = tot_eval_ns = 0
        tool_trace: List[Dict[str, Any]] = []
        final_text = ""
        turns = 0

        for turn in range(1, max_turns + 1):
            turns = turn
            if _STOP.is_set():
                break
            resp = client.chat(model, messages, tools=NATIVE_TOOLS, options=options)
            msg = resp.get("message", {}) or {}
            tot_prompt += int(resp.get("prompt_eval_count", 0) or 0)
            tot_eval += int(resp.get("eval_count", 0) or 0)
            tot_dur_ns += int(resp.get("total_duration", 0) or 0)
            tot_eval_ns += int(resp.get("eval_duration", 0) or 0)

            calls = msg.get("tool_calls")
            if calls:
                assistant: Dict[str, Any] = {"role": "assistant", "content": msg.get("content", "")}
                assistant["tool_calls"] = calls
                messages.append(assistant)
                for call in calls:
                    fn = (call.get("function") or {})
                    tname = fn.get("name", "")
                    targs = _coerce_args(fn.get("arguments"))
                    result = execute_tool(tname, targs, workspace, tool_timeout)
                    tool_trace.append({"turn": turn, "tool": tname, "args": targs,
                                       "result": result[:500]})
                    if verbose:
                        print(f"    [agent t{turn}] {tname}({_summarize_args(targs)})")
                    messages.append({"role": "tool", "name": tname, "content": result})
            else:
                final_text = msg.get("content", "") or ""
                break

        stats = GenerationStats()
        stats.prompt_count = tot_prompt
        stats.eval_count = tot_eval
        stats.total_duration_s = tot_dur_ns / 1e9
        stats.prompt_eval_rate_tps = (tot_prompt / ((tot_dur_ns - tot_eval_ns) / 1e9)
                                     if (tot_dur_ns - tot_eval_ns) > 0 else 0.0)
        stats.eval_rate_tps = (tot_eval / (tot_eval_ns / 1e9)) if tot_eval_ns > 0 else 0.0
        return {"final_text": final_text, "stats": stats, "turns": turns,
                "tool_trace": tool_trace, "workspace": workspace}
    except Exception:
        raise


def evaluate_native_tool(task: Task, workspace: str) -> Dict[str, Any]:
    """Run the in-process ``native_verify`` snippet against the workspace."""
    ws = WorkspaceView(workspace)
    ns: Dict[str, Any] = {"ws": ws}
    code = task.native_verify
    if not code or not code.strip():
        return {"success": False, "error": "no native_verify defined", "score": 0.0,
                "passed": 0, "total": 1}
    try:
        exec(compile(code, f"<verify:{task.id}>", "exec"), ns)  # noqa: S102
        success = bool(ns.get("success", False))
        message = str(ns.get("message", ""))
    except Exception as e:  # noqa: BLE001
        success = False
        message = f"verifier error: {e!r}"
    return {
        "success": success,
        "error": "" if success else message,
        "score": 100.0 if success else 0.0,
        "passed": 1 if success else 0,
        "total": 1,
        "message": message[:400],
    }
# ----------------------------------------------------------------------------
# Single-run executor (error isolation, power capture, native/proxy/cli)
# ----------------------------------------------------------------------------
def run_ollama(
    model: str,
    task: Task,
    client: Optional["OllamaClient"],
    backend: str,
    options: Dict[str, Any],
    timeout: int,
    tool_mode: str,
    max_agent_turns: int,
    verbose: bool = False,
) -> RunResult:
    """Run one (model, task) pair. Never raises: failures become a failed RunResult."""
    res = RunResult(
        model=model, task_id=task.id, task_category=task.category,
        eval_type=task.eval_type, language=task.language, difficulty=task.difficulty,
        response_text="", success=False, error="", score=0.0, passed=0, total=1,
        backend=backend, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    if _STOP.is_set():
        res.completed = False
        res.error = "stopped by user"
        return res

    use_native = (task.eval_type == "tool" and tool_mode == "native"
                  and backend == "api" and client is not None)

    monitor = PowerMonitor() if nvidia_smi_available() else None
    if monitor:
        monitor.start()
    generation_failed = False
    workspace: Optional[str] = None
    try:
        t0 = time.monotonic()
        if use_native:
            agent = run_native_agent(model, task, client, options,
                                     max_turns=max_agent_turns, verbose=verbose)
            workspace = agent["workspace"]
            res.response_text = agent["final_text"]
            res.stats = agent["stats"]
            res.backend = "api:native"
        elif backend == "api" and client is not None:
            res.response_text, res.stats = client.generate(model, task.prompt, options)
        else:
            text, stats, err = run_ollama_cli(model, task.prompt, timeout)
            res.response_text = text
            res.stats = stats
            if err and not text:
                generation_failed = True
                res.error = err
        res.latency_ms = (time.monotonic() - t0) * 1000.0

        if not generation_failed:
            if use_native:
                ev = evaluate_native_tool(task, workspace)
            elif task.eval_type == "tool":
                ev = evaluate_tool_json(task, res.response_text)
            else:
                ev = evaluate_code_response(task, res.response_text)
            res.success = bool(ev.get("success"))
            res.score = float(ev.get("score", 0.0))
            res.passed = int(ev.get("passed", 0))
            res.total = int(ev.get("total", 1))
            res.error = "" if res.success else str(ev.get("error", ""))[:500]
    except OllamaError as e:
        generation_failed = True
        res.error = f"API error: {e}"
    except Exception as e:  # noqa: BLE001 - isolate any per-run failure
        generation_failed = True
        res.error = f"unexpected error: {e!r}"
    finally:
        if monitor:
            res.power = monitor.stop()
    return res
# ----------------------------------------------------------------------------
# Result (de)serialization + checkpointing
# ----------------------------------------------------------------------------
_GEN_STATS_FIELDS = frozenset(GenerationStats.__dataclass_fields__)
_RUN_RESULT_FIELDS = frozenset(RunResult.__dataclass_fields__)


def run_result_to_dict(rr: RunResult) -> Dict[str, Any]:
    """Serialize a RunResult to a JSON-friendly dict (nested stats -> dict)."""
    return asdict(rr)


def run_result_from_dict(d: Dict[str, Any]) -> RunResult:
    """Rebuild a RunResult from a dict, ignoring unknown keys (forward-safe)."""
    d = dict(d or {})
    stats_in = d.pop("stats", None)
    if isinstance(stats_in, GenerationStats):
        stats_obj = stats_in
    else:
        stats_in = stats_in or {}
        stats_obj = GenerationStats(**{k: v for k, v in stats_in.items() if k in _GEN_STATS_FIELDS})
    kwargs = {k: v for k, v in d.items() if k in _RUN_RESULT_FIELDS}
    kwargs["stats"] = stats_obj
    return RunResult(**kwargs)


def _ckpt_path(output_dir: str) -> str:
    return os.path.join(output_dir, "benchmark.ckpt.json")


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a checkpoint file; return {} if absent or malformed."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {}


def save_checkpoint(path: str, ckpt: Dict[str, Any]) -> None:
    """Atomically write a checkpoint (temp file + os.replace)."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _work_key(model: str, task: Task, it: int) -> str:
    """Stable identity for a single (model, task, iteration) unit of work."""
    return f"{model}|{task.id}|{it}"
def _progress(log, done: int, total: int, rr: RunResult, verbose: bool) -> None:
    status = "PASS" if rr.success else "FAIL"
    line = (f"  [{done:>3}/{total}] {rr.model:30} {rr.task_id:24} "
            f"{status:4} score={rr.score:5.1f} {rr.latency_ms:9.0f}ms")
    if verbose and rr.error:
        line += f"  | {rr.error[:90]}"
    log(line)


# ----------------------------------------------------------------------------
# Benchmark driver (ordered work, thread pool, checkpointing, Ctrl-C safe)
# ----------------------------------------------------------------------------
def run_benchmark(
    models: Sequence[str],
    tasks: Sequence[Task],
    client: Optional[OllamaClient],
    backend: str,
    options: Dict[str, Any],
    timeout: int,
    tool_mode: str,
    max_agent_turns: int,
    iterations: int,
    workers: int,
    output_dir: str,
    resume: bool = False,
    verbose: bool = False,
    quiet: bool = False,
) -> List[RunResult]:
    """Run every (model, task, iteration) and return an ordered list of results.

    - On ``--resume``, already-completed units are skipped (loaded from the
      checkpoint) and the rest are filled in.
    - Results are checkpointed after each unit so a Ctrl-C never loses progress.
    - Returns results in the deterministic model -> task -> iteration order.
    """
    t_start = time.time()

    def log(msg: str) -> None:
        if not quiet:
            print(msg, file=sys.stderr, flush=True)

    iters = max(1, int(iterations))
    work = [(m, t, it) for m in models for t in tasks for it in range(1, iters + 1)]
    total = len(work)
    if total == 0:
        log("[benchmark] nothing to do (no models or tasks)")
        return []

    cpath = _ckpt_path(output_dir)
    ckpt: Dict[str, Any] = load_checkpoint(cpath) if resume else {}
    results: Dict[str, Dict[str, Any]] = dict(ckpt.get("results", {}))

    meta: Dict[str, Any] = {
        "models": list(models), "backend": backend, "tool_mode": tool_mode,
        "options": options, "timeout": timeout, "iterations": iters,
        "workers": workers, "max_agent_turns": max_agent_turns,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if resume and isinstance(ckpt.get("meta"), dict):
        meta["started"] = ckpt["meta"].get("started", meta["started"])

    pending = [u for u in work if _work_key(*u) not in results]
    log(f"[benchmark] {total} runs total | {total - len(pending)} already done | "
        f"{len(pending)} to run | backend={backend} | workers={workers}")

    def do_run(model: str, task: Task, it: int) -> Tuple[str, RunResult]:
        return _work_key(model, task, it), run_ollama(
            model, task, client, backend, options, timeout, tool_mode,
            max_agent_turns, verbose,
        )

    def record(k: str, rr: RunResult) -> None:
        with _results_lock:
            results[k] = run_result_to_dict(rr)
            done = len(results)
            save_checkpoint(cpath, {
                "version": "2",
                "meta": meta,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "results": results,
            })
        _progress(log, done, total, rr, verbose)

    if workers and workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(do_run, *u) for u in pending]
            for fut in as_completed(futures):
                k, rr = fut.result()
                record(k, rr)
    else:
        for u in pending:
            if _STOP.is_set():
                log("[benchmark] stop requested - not scheduling any further runs")
                break
            record(*do_run(*u))

    ordered: List[RunResult] = []
    for u in work:
        kd = _work_key(*u)
        if kd in results:
            ordered.append(run_result_from_dict(results[kd]))

    log(f"[benchmark] finished in {time.time() - t_start:.1f}s -> {len(ordered)} results")
    return ordered
# ----------------------------------------------------------------------------
# Scoring / aggregation
# ----------------------------------------------------------------------------
#: Default weights for the composite score (renormalized when a dimension is
#: not measurable for a model, e.g. no tool tasks or no GPU power data).
COMPOSITE_WEIGHTS = {"correctness": 0.4, "agentic": 0.2, "speed": 0.2, "power": 0.2}


def _weighted_score(results: Sequence[RunResult]) -> float:
    """Difficulty-weighted mean of per-task scores (0-100)."""
    w = sum(r.difficulty for r in results) or 1.0
    return sum(r.score * r.difficulty for r in results) / w


def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _std(vals: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return statistics.pstdev(vals) if len(vals) >= 2 else 0.0


def _energy_joules(rr: RunResult) -> Optional[float]:
    """Approximate energy (J) = avg GPU power (W) * total generation duration (s)."""
    avg_w = (rr.power or {}).get("gpu_power_avg_w")
    dur = rr.stats.total_duration_s
    if avg_w and dur:
        return avg_w * dur
    return None


def _round(v: Optional[float], nd: int = 2) -> Optional[float]:
    return round(v, nd) if v is not None else None


def normalize_relative(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Normalize each value to [0,100] relative to the batch maximum (best=100)."""
    finite = [v for v in values if v is not None and v > 0]
    if not finite:
        return [0.0 if v is not None else None for v in values]
    mx = max(finite)
    return [(v / mx * 100.0) if (v is not None and v > 0) else 0.0 for v in values]


def normalize_absolute(value: Optional[float], ref: Optional[float]) -> Optional[float]:
    """Normalize a single value to [0,100] against an absolute reference (capped)."""
    if value is None:
        return None
    if not ref or ref <= 0:
        return 0.0
    return min(100.0, value / ref * 100.0)


def composite_score(
    components: Dict[str, Optional[float]],
    weights: Dict[str, float] = COMPOSITE_WEIGHTS,
) -> Optional[float]:
    """Weighted mean of the available 0-100 components (weights renormalized)."""
    num = den = 0.0
    for k, w in weights.items():
        v = components.get(k)
        if v is None:
            continue
        num += w * v
        den += w
    return (num / den) if den > 0 else None
def summarize_models(
    results: Sequence[RunResult],
    normalize: str = "relative",
    ref_rate: Optional[float] = None,
    ref_tpj: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Aggregate per-model results into a ranked summary list (best first)."""
    order: List[str] = []
    by_model: Dict[str, List[RunResult]] = {}
    for rr in results:
        if rr.model not in by_model:
            by_model[rr.model] = []
            order.append(rr.model)
        by_model[rr.model].append(rr)

    raw: Dict[str, Dict[str, Any]] = {}
    for model in order:
        rs = by_model[model]
        code = [r for r in rs if r.eval_type == "code"]
        tool = [r for r in rs if r.eval_type == "tool"]
        speeds = [r.stats.eval_rate_tps for r in rs if r.stats.eval_rate_tps > 0]
        tpj = []
        for r in rs:
            e = _energy_joules(r)
            if e and r.stats.eval_count > 0:
                tpj.append(r.stats.eval_count / e)
        lat = [r.latency_ms for r in rs if r.latency_ms > 0]
        raw[model] = {
            "runs": len(rs),
            "pass_rate": (sum(1 for r in rs if r.success) / len(rs)) * 100.0,
            "correctness": _weighted_score(code) if code else None,
            "agentic": _weighted_score(tool) if tool else None,
            "speed_tps": _mean(speeds),
            "tpj": _mean(tpj),
            "lat_mean": _mean(lat),
            "lat_std": _std(lat),
            "scores": [r.score for r in rs],
        }

    speed_raw = [raw[m]["speed_tps"] for m in order]
    tpj_raw = [raw[m]["tpj"] for m in order]
    if normalize == "absolute":
        speed_norm = [normalize_absolute(v, ref_rate) for v in speed_raw]
        power_norm = [normalize_absolute(v, ref_tpj) for v in tpj_raw]
    else:
        speed_norm = normalize_relative(speed_raw)
        power_norm = normalize_relative(tpj_raw)

    summary: List[Dict[str, Any]] = []
    for i, m in enumerate(order):
        r = raw[m]
        comps = {"correctness": r["correctness"], "agentic": r["agentic"],
                 "speed": speed_norm[i], "power": power_norm[i]}
        scores = r["scores"]
        summary.append({
            "model": m,
            "runs": r["runs"],
            "pass_rate": round(r["pass_rate"], 1),
            "correctness": _round(r["correctness"]),
            "agentic": _round(r["agentic"]),
            "speed_tps": _round(r["speed_tps"], 2),
            "tpj": _round(r["tpj"], 4),
            "speed_norm": _round(speed_norm[i]),
            "power_norm": _round(power_norm[i]),
            "composite": _round(composite_score(comps)),
            "latency_ms_mean": _round(r["lat_mean"], 1),
            "latency_ms_std": _round(r["lat_std"], 1),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "score_median": _round(statistics.median(scores)) if scores else None,
            "score_std": _round(_std(scores)),
        })
    summary.sort(key=lambda s: (s["composite"] is None, -(s["composite"] or 0.0)))
    for rank, s in enumerate(summary, 1):
        s["rank"] = rank
    return summary
# ----------------------------------------------------------------------------
# Reporting: JSON + CSV
# ----------------------------------------------------------------------------
def build_report(
    results: Sequence[RunResult],
    config: Dict[str, Any],
    normalize: str = "relative",
    ref_rate: Optional[float] = None,
    ref_tpj: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble the full, JSON-serializable report object."""
    summary = summarize_models(results, normalize, ref_rate, ref_tpj)
    return {
        "version": "2.0",
        "tool": "ollama-coding-benchmark",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": get_system_info(),
        "config": config,
        "summary": summary,
        "results": [run_result_to_dict(r) for r in results],
    }


def write_json(report: Dict[str, Any], path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


CSV_FIELDS = [
    "model", "task_id", "task_category", "eval_type", "language", "difficulty",
    "success", "score", "passed", "total", "backend", "latency_ms",
    "prompt_count", "eval_count", "prompt_eval_rate_tps", "eval_rate_tps",
    "total_duration_s", "ttft_s", "gpu_power_avg_w", "timestamp", "error",
]


def _csv_row(rr: RunResult) -> List[Any]:
    st = rr.stats
    p = rr.power or {}
    return [
        rr.model, rr.task_id, rr.task_category, rr.eval_type, rr.language, rr.difficulty,
        int(bool(rr.success)), rr.score, rr.passed, rr.total, rr.backend, rr.latency_ms,
        st.prompt_count, st.eval_count, st.prompt_eval_rate_tps, st.eval_rate_tps,
        st.total_duration_s, st.ttft_s, p.get("gpu_power_avg_w", ""), rr.timestamp, rr.error,
    ]


def write_csv(results: Sequence[RunResult], path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for rr in results:
            w.writerow(_csv_row(rr))
def _md_cell(v: Any) -> str:
    s = str(v) if v is not None else ""
    return s.replace("|", "\\|").replace("\n", " ")[:140]


def _fmt(v: Optional[float], nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_markdown(report: Dict[str, Any], path: str) -> None:
    cfg = report.get("config", {}) or {}
    L: List[str] = []
    L += [
        "# Ollama Coding Benchmark",
        "",
        f"- **Generated:** {report.get('generated_at')}  ",
        f"- **Version:** {report.get('version')}  ",
        f"- **Models:** {_md_cell(', '.join(cfg.get('models', [])))}  ",
        f"- **Backend / tool mode:** {cfg.get('backend')} / {cfg.get('tool_mode')}  ",
        f"- **Options:** `{json.dumps(cfg.get('options', {}))}`  ",
        f"- **Normalization:** {cfg.get('normalize', 'relative')}",
        "",
        "## Environment",
        "",
        "| Component | Value |",
        "|---|---|",
    ]
    sysinfo = report.get("system", {}) or {}
    for k in ("os", "cpu", "memory", "ollama_version", "python"):
        if sysinfo.get(k) is not None:
            L.append(f"| {k} | {_md_cell(sysinfo[k])} |")
    if sysinfo.get("gpus"):
        L.append(f"| gpu | {_md_cell(' ; '.join(str(g) for g in sysinfo['gpus']))} |")
    L += [
        "",
        "## Leaderboard",
        "",
        "| # | Model | Composite | Correctness | Agentic | Speed (tok/s) | Power (tok/J) | Pass rate | Runs |",
        "|--:|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for s in report.get("summary", []):
        L.append("| {} | {} | {} | {} | {} | {} | {} | {}% | {} |".format(
            s.get("rank"), _md_cell(s.get("model")), _fmt(s.get("composite")),
            _fmt(s.get("correctness")), _fmt(s.get("agentic")),
            _fmt(s.get("speed_tps")), _fmt(s.get("tpj")),
            _fmt(s.get("pass_rate"), 1), s.get("runs")))
    L += [
        "",
        "## Per-task breakdown",
        "",
        "| Model | Task | Type | Score | Passed | Backend | Error |",
        "|---|---|--:|--:|---|---|",
    ]
    for r in report.get("results", []):
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            _md_cell(r.get("model")), _md_cell(r.get("task_id")), r.get("eval_type"),
            _fmt(r.get("score"), 1), f'{r.get("passed")}/{r.get("total")}',
            _md_cell(r.get("backend")), _md_cell(r.get("error") or "")))
    L.append("")
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def write_reports(report: Dict[str, Any], results: Sequence[RunResult],
                  output_dir: str) -> List[str]:
    """Write JSON + CSV + Markdown reports; return the list of file paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = [
        os.path.join(output_dir, "report.json"),
        os.path.join(output_dir, "results.csv"),
        os.path.join(output_dir, "report.md"),
    ]
    write_json(report, paths[0])
    write_csv(results, paths[1])
    write_markdown(report, paths[2])
    return paths
# ----------------------------------------------------------------------------
# Console leaderboard + cross-run comparison
# ----------------------------------------------------------------------------
def print_leaderboard(summary: Sequence[Dict[str, Any]], title: str = "Leaderboard") -> None:
    if not summary:
        print("\n  (no results)")
        return
    print(f"\n  {title}")
    print("  " + "-" * 80)
    print("  {:>3}  {:<24} {:>7} {:>6} {:>5} {:>10} {:>8} {:>6} {:>5}".format(
        "#", "Model", "Comp", "Corr", "Agen", "Speed t/s", "Power", "Pass%", "Runs"))
    print("  " + "-" * 80)

    def g(row: Dict[str, Any], k: str, nd: int = 1) -> str:
        v = row.get(k)
        return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    for s in summary:
        print("  {:>3}  {:<24} {:>7} {:>6} {:>5} {:>10} {:>8} {:>6} {:>5}".format(
            s.get("rank"), (s.get("model") or "")[:24],
            g(s, "composite"), g(s, "correctness"), g(s, "agentic"),
            g(s, "speed_tps", 2), g(s, "tpj", 3), g(s, "pass_rate", 1), s.get("runs")))
    print("  " + "-" * 80)


def _load_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_runs(path_a: str, path_b: str) -> None:
    """Print a per-dimension delta (B - A) table for two saved JSON reports."""
    ra = _load_report(path_a)
    rb = _load_report(path_b)
    sa = {s.get("model"): s for s in ra.get("summary", [])}
    sb = {s.get("model"): s for s in rb.get("summary", [])}
    order = list(sa.keys()) + [m for m in sb if m not in sa]
    if not order:
        print("\n  (no comparable models in the two reports)")
        return
    dims = [("composite", 1), ("correctness", 1), ("agentic", 1),
            ("speed_tps", 2), ("tpj", 3), ("pass_rate", 1)]
    print(f"\n  Comparing  A={os.path.basename(path_a)}   B={os.path.basename(path_b)}")
    print("  (values are the delta B - A)")
    print("  " + "-" * 78)
    print("  {:<24}".format("Model") + "".join("{:>10}".format(d) for d, _ in dims))
    print("  " + "-" * 78)
    for m in order:
        a = sa.get(m, {}) or {}
        b = sb.get(m, {}) or {}
        line = "  {:<24}".format(m[:24])
        for d, nd in dims:
            av, bv = a.get(d), b.get(d)
            if not isinstance(av, (int, float)) and not isinstance(bv, (int, float)):
                line += "{:>10}".format("-")
                continue
            avv = float(av) if isinstance(av, (int, float)) else 0.0
            bvv = float(bv) if isinstance(bv, (int, float)) else 0.0
            line += "{:>10}".format(f"{bvv - avv:+.{nd}f}")
        print(line)
    print("  " + "-" * 78)
# ----------------------------------------------------------------------------
# Task filtering + listing
# ----------------------------------------------------------------------------
def filter_tasks(
    tasks: Sequence[Task],
    categories: Optional[Sequence[str]] = None,
    difficulty: Optional[str] = None,
    language: Optional[str] = None,
) -> List[Task]:
    """Filter the task set by category, difficulty label and language."""
    out: List[Task] = []
    for t in tasks:
        if categories and t.category not in categories:
            continue
        if (language and language.lower() != "all"
                and t.eval_type == "code" and t.language.lower() != language.lower()):
            continue
        if (difficulty and difficulty.lower() != "all"
                and difficulty_label(t.difficulty) != difficulty.lower()):
            continue
        out.append(t)
    return out


def _print_task_list(tasks: Sequence[Task]) -> None:
    print(f"\n  Built-in tasks ({len(tasks)}):")
    print("  " + "-" * 72)
    print("  {:<24} {:<12} {:<6} {:<10} {:<7} {}".format(
        "id", "category", "type", "language", "level", "notes"))
    print("  " + "-" * 72)
    for t in tasks:
        notes = []
        if t.native_prompt or t.native_files or t.native_verify:
            notes.append("native")
        if t.required_tools:
            notes.append("tools=" + ",".join(t.required_tools))
        print("  {:<24} {:<12} {:<6} {:<10} {:<7} {}".format(
            t.id[:24], t.category[:12], t.eval_type, t.language,
            difficulty_label(t.difficulty), " ".join(notes)))
    print("  " + "-" * 72)
# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ollama_coding_benchmark",
        description="Benchmark Ollama models on coding, agentic, speed and power dimensions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Targets
    p.add_argument("--models", nargs="+", default=["llama3.1:8b"],
                   help="Model name(s) to benchmark.")
    p.add_argument("--task-categories", nargs="+", default=None,
                   help="Only run tasks in these categories (e.g. Python JavaScript Agentic).")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"], default="all",
                   help="Only run tasks of this difficulty.")
    p.add_argument("--language", choices=["python", "javascript", "bash", "all"], default="all",
                   help="Only run code tasks in this language.")
    # Sampling / generation
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    p.add_argument("--seed", type=int, default=None, help="Random seed (reproducible sampling).")
    p.add_argument("--num-ctx", type=int, default=None, help="Context window size (num_ctx).")
    p.add_argument("--num-predict", type=int, default=None, help="Max tokens to generate (num_predict).")
    p.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    p.add_argument("--top-p", type=float, default=None, help="Top-p sampling.")
    # Execution
    p.add_argument("--backend", choices=["api", "cli"], default="api",
                   help="Ollama HTTP API or `ollama run` CLI fallback.")
    p.add_argument("--host", default=None,
                   help="Ollama host (default: $OLLAMA_HOST or http://127.0.0.1:11434).")
    p.add_argument("--timeout", type=int, default=300, help="Per-request timeout in seconds.")
    p.add_argument("--tool-mode", choices=["proxy", "native"], default="proxy",
                   help="Tool-use mode: one-shot proxy or real native tool-calling agent loop.")
    p.add_argument("--max-agent-turns", type=int, default=6, help="Max agent loop turns (native mode).")
    p.add_argument("--iterations", type=int, default=1, help="Independent iterations per model/task.")
    p.add_argument("--workers", type=int, default=1, help="Concurrent workers (1 = serial).")
    # Scoring / normalization
    p.add_argument("--normalize", choices=["relative", "absolute"], default="relative",
                   help="Speed/power normalization: relative to batch max or absolute reference.")
    p.add_argument("--ref-rate", type=float, default=None, help="Absolute ref for speed (tok/s).")
    p.add_argument("--ref-tpj", type=float, default=None, help="Absolute ref for power (tok/joule).")
    # Output / robustness
    p.add_argument("--output-dir", default="outputs", help="Directory for reports + checkpoint.")
    p.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint.")
    p.add_argument("--list-tasks", action="store_true", help="Print the built-in task set and exit.")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None,
                   help="Compare two report.json files (delta B - A) and exit.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-run progress output.")
    p.add_argument("--verbose", action="store_true", help="Verbose per-run / agent output.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)
def _model_matches(wanted: str, available: str) -> bool:
    if wanted == available:
        return True
    base = available.split(":")[0]
    return wanted == base or available.startswith(wanted)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        signal.signal(signal.SIGINT, _request_stop)
    except (ValueError, OSError):
        pass  # not on the main thread - skip the graceful-stop hook

    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
        return 0

    if args.list_tasks:
        _print_task_list(TASKS)
        return 0

    tasks = filter_tasks(TASKS, args.task_categories, args.difficulty, args.language)
    if not tasks:
        print("error: no tasks match the given filters", file=sys.stderr)
        return 2

    options = build_generation_options(
        temperature=args.temperature, seed=args.seed, num_ctx=args.num_ctx,
        num_predict=args.num_predict, top_k=args.top_k, top_p=args.top_p,
    )

    client = OllamaClient(host=args.host, timeout=args.timeout) if args.backend == "api" else None

    if args.backend == "api" and client is not None:
        if not client.is_available():
            print(f"error: Ollama server not reachable at {client.host}\n"
                  "       (start it with `ollama serve`, or use --backend cli)", file=sys.stderr)
            return 3
        available = client.list_models()
        for name in args.models:
            if available and not any(_model_matches(name, a) for a in available):
                print(f"[warn] model '{name}' not found on server "
                      f"(available: {', '.join(available) or 'none'})", file=sys.stderr)

    config: Dict[str, Any] = {
        "models": list(args.models), "backend": args.backend, "tool_mode": args.tool_mode,
        "options": options, "timeout": args.timeout, "iterations": args.iterations,
        "workers": args.workers, "max_agent_turns": args.max_agent_turns,
        "normalize": args.normalize, "ref_rate": args.ref_rate, "ref_tpj": args.ref_tpj,
        "host": (client.host if client else None),
    }

    results = run_benchmark(
        models=args.models, tasks=tasks, client=client, backend=args.backend,
        options=options, timeout=args.timeout, tool_mode=args.tool_mode,
        max_agent_turns=args.max_agent_turns, iterations=args.iterations,
        workers=args.workers, output_dir=args.output_dir, resume=args.resume,
        verbose=args.verbose, quiet=args.quiet,
    )

    report = build_report(results, config, args.normalize, args.ref_rate, args.ref_tpj)
    paths = write_reports(report, results, args.output_dir)
    print_leaderboard(report["summary"])
    if not args.quiet:
        print("\n  Wrote:")
        for path in paths:
            print(f"    - {path}")
        print(f"    - {_ckpt_path(args.output_dir)}  (checkpoint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
