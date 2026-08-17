"""Behavioral tests for the built-in task definitions and evaluation harnesses.

Every *code* task must be solvable by a known-correct reference implementation,
and deliberately-wrong implementations must be rejected. These tests are
deterministic and need no running Ollama server — they exercise the Python /
JavaScript / bash code harnesses and the tool-JSON scorer against ``TASKS``.

Run directly (dependency-free)::

    python tests/test_task_definitions.py

Or via pytest::

    pytest tests/
"""
from __future__ import annotations

import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ollama_coding_benchmark as m  # noqa: E402

BY_ID = {t.id: t for t in m.TASKS}

# --- Reference implementations (known-correct) -------------------------------
REF: dict = {}
REF["py_reverse_string"] = "def reverse_string(s):\n    return s[::-1]\n"
REF["py_fizzbuzz"] = (
    "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n"
    "        if i % 15 == 0:\n            out.append('FizzBuzz')\n"
    "        elif i % 3 == 0:\n            out.append('Fizz')\n"
    "        elif i % 5 == 0:\n            out.append('Buzz')\n"
    "        else:\n            out.append(str(i))\n    return out\n"
)
REF["py_is_palindrome"] = (
    "def is_palindrome(s):\n    c = [ch.lower() for ch in s if ch.isalnum()]\n"
    "    return c == c[::-1]\n"
)
REF["py_count_vowels"] = (
    "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')\n"
)
REF["py_matrix_transpose"] = (
    "def matrix_transpose(m):\n"
    "    return [[row[i] for row in m] for i in range(len(m[0]))] if m and m[0] else []\n"
)
REF["py_binary_search"] = (
    "def binary_search(nums, target):\n    lo, hi = 0, len(nums) - 1\n"
    "    while lo <= hi:\n        mid = (lo + hi) // 2\n"
    "        if nums[mid] == target:\n            return mid\n"
    "        elif nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n"
    "    return -1\n"
)
REF["py_two_sum"] = (
    "def two_sum(nums, target):\n    seen = {}\n"
    "    for i, x in enumerate(nums):\n        need = target - x\n"
    "        if need in seen:\n            return sorted([seen[need], i])\n"
    "        seen[x] = i\n    return []\n"
)
REF["py_word_frequency"] = (
    "from collections import Counter\n"
    "def word_frequency(text, top_n=0):\n"
    "    c = Counter(w.lower() for w in text.split() if w)\n"
    "    items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))\n"
    "    return items[:top_n] if top_n > 0 else items\n"
)
REF["py_merge_intervals"] = (
    "def merge_intervals(intervals):\n"
    "    if not intervals:\n        return []\n"
    "    ivs = sorted(intervals)\n    merged = [list(ivs[0])]\n"
    "    for s, e in ivs[1:]:\n"
    "        if s <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], e)\n"
    "        else:\n            merged.append([s, e])\n"
    "    return merged\n"
)
REF["py_email_extract"] = (
    "import re\n"
    "def extract_emails(text):\n"
    "    return re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}', text)\n"
)
REF["py_lru_cache"] = (
    "from collections import OrderedDict\n"
    "class LRUCache:\n"
    "    def __init__(self, capacity):\n        self.cap = capacity\n        self.d = OrderedDict()\n"
    "    def get(self, key):\n"
    "        if key not in self.d:\n            return -1\n"
    "        self.d.move_to_end(key)\n        return self.d[key]\n"
    "    def put(self, key, value):\n"
    "        if key in self.d:\n            self.d.move_to_end(key)\n"
    "        self.d[key] = value\n"
    "        if len(self.d) > self.cap:\n            self.d.popitem(last=False)\n"
)
REF["py_bfs_shortest_path"] = (
    "from collections import deque\n"
    "def shortest_path(graph, start, end):\n"
    "    if start == end:\n        return 0\n"
    "    seen = {start}\n    q = deque([(start, 0)])\n"
    "    while q:\n        node, d = q.popleft()\n"
    "        for nb in graph.get(node, []):\n"
    "            if nb == end:\n                return d + 1\n"
    "            if nb not in seen:\n                seen.add(nb)\n                q.append((nb, d + 1))\n"
    "    return -1\n"
)
REF["py_token_bucket"] = (
    "class TokenBucket:\n"
    "    def __init__(self, capacity, rate):\n"
    "        self.cap = capacity\n        self.rate = rate\n"
    "        self.tokens = capacity\n        self.last = 0.0\n"
    "    def try_acquire(self, tokens=1, now=0.0):\n"
    "        elapsed = max(0.0, now - self.last)\n"
    "        self.tokens = min(self.cap, self.tokens + elapsed * self.rate)\n"
    "        self.last = now\n"
    "        if self.tokens >= tokens:\n            self.tokens -= tokens\n            return True\n"
    "        return False\n"
)
# JavaScript: group array items by a key property (order preserved).
REF["js_group_by"] = (
    "function groupBy(arr, key) {\n"
    "  const out = {};\n"
    "  for (const item of arr) {\n"
    "    const k = item[key];\n"
    "    (out[k] = out[k] || []).push(item);\n"
    "  }\n"
    "  return out;\n"
    "}\n"
)
# JavaScript: count lowercase vowels/consonants; uppercase is ignored.
REF["js_string_stats"] = (
    "function stringStats(s) {\n"
    "  let vowels = 0, consonants = 0;\n"
    "  for (const ch of s) {\n"
    "    if ('aeiou'.includes(ch)) vowels++;\n"
    "    else if (ch >= 'a' && ch <= 'z') consonants++;\n"
    "  }\n"
    "  return { vowels, consonants };\n"
    "}\n"
)
REF["bash_count_lines"] = "count_lines() { wc -l < \"$1\"; }\n"


def _code_eval(task_id: str, language: str, code: str) -> dict:
    return m.evaluate_code_response(BY_ID[task_id], f"```{language}\n{code}\n```")


def test_python_reference_solutions() -> None:
    checked = 0
    for t in m.TASKS:
        if t.eval_type != "code" or t.language != "python":
            continue
        checked += 1
        ev = _code_eval(t.id, "python", REF[t.id])
        assert ev["success"], f"{t.id}: {ev}"
        assert ev["passed"] == ev["total"], f"{t.id}: {ev}"
    assert checked > 0, "no python code tasks found"


def test_javascript_reference_solutions() -> None:
    if shutil.which("node") is None:
        return  # skip when node is unavailable
    for t in m.TASKS:
        if t.eval_type != "code" or t.language != "javascript":
            continue
        ev = _code_eval(t.id, "javascript", REF[t.id])
        assert ev["success"], f"{t.id}: {ev}"


def test_bash_reference_solutions() -> None:
    if shutil.which("bash") is None:
        return  # skip when bash is unavailable
    for t in m.TASKS:
        if t.eval_type != "code" or t.language != "bash":
            continue
        ev = _code_eval(t.id, "bash", REF[t.id])
        assert ev["success"], f"{t.id}: {ev}"


def test_tool_reference_plans() -> None:
    for t in m.TASKS:
        if t.eval_type != "tool" or not t.id.startswith("tool_"):
            continue
        ans = "```json\n" + json.dumps(t.required_tools) + "\n```"
        ev = m.evaluate_tool_json(t, ans)
        assert ev["success"], f"{t.id}: {ev}"


def test_wrong_solutions_are_rejected() -> None:
    bad = [
        ("py_reverse_string", "python", "def reverse_string(s):\n    return s\n"),
        ("py_fizzbuzz", "python", "def fizzbuzz(n):\n    return ['x'] * n\n"),
        ("py_two_sum", "python", "def two_sum(nums, target):\n    return [0, 0]\n"),
    ]
    for tid, lang, code in bad:
        ev = _code_eval(tid, lang, code)
        assert not ev["success"], f"{tid}: expected a failing score, got {ev}"
    # A tool plan missing required tools must not score fully.
    t = BY_ID["tool_send_email"]
    ev = m.evaluate_tool_json(t, "```json\n[\"email\"]\n```")
    assert not ev["success"], f"tool_send_email: expected a failing score, got {ev}"


def test_thermal_throttle_detection() -> None:
    d = m.detect_thermal_throttle
    # Hot + sustained power drop under continued load => throttled.
    powers = [200.0] * 20 + [120.0] * 20
    temps_hot = [70.0] * 20 + [85.0] * 20
    r = d(powers, temps_hot, threshold_c=83.0)
    assert r["thermal_throttled"] is True, r
    assert r["thermal_throttle_events"] >= 1, r
    assert (r["thermal_power_drop_pct"] or 0.0) >= 30.0, r

    # Same power drop but the card is COOL (end-of-gen idle dip) => not throttled.
    temps_cool = [70.0] * 20 + [45.0] * 20
    assert d(powers, temps_cool, threshold_c=83.0)["thermal_throttled"] is False

    # Hot but power stays high (no drop) => not throttled.
    assert d([200.0] * 40, [85.0] * 40, threshold_c=83.0)["thermal_throttled"] is False

    # Below the sample floor => not throttled (not enough data).
    assert d([200.0, 100.0], [85.0, 85.0], threshold_c=83.0)["thermal_throttled"] is False


def test_discover_models() -> None:
    sample = (
        "NAME           ID              SIZE      MODIFIED\n"
        "llama3.1:8b    46e0c10c039e    4.9 GB    2 months ago\n"
        "qwen3:4b       abc123def0      2.5 GB    1 month ago\n"
    )
    assert m._parse_ollama_list(sample) == ["llama3.1:8b", "qwen3:4b"]
    assert m._parse_ollama_list("NAME  ID  SIZE\n") == []
    assert m._parse_ollama_list("") == []

    class _StubClient:
        def list_models(self):
            return ["m1", "m2"]

    assert m._discover_models(_StubClient(), "api") == ["m1", "m2"]


def test_gpu_cool_down() -> None:
    def _silence(msg: str) -> None:
        pass

    # _parse_gpu_temps: hottest value wins; non-numeric tokens ignored.
    assert m._parse_gpu_temps("45\n60\n") == 60.0
    assert m._parse_gpu_temps("55") == 55.0
    assert m._parse_gpu_temps("") is None
    assert m._parse_gpu_temps("n/a\n") is None

    # wait_for_gpu_cool_down: no-op when already cool, disabled, or no GPU.
    orig = m.current_gpu_temp
    try:
        m.current_gpu_temp = lambda: 50.0
        assert m.wait_for_gpu_cool_down(80.0, 8.0, log=_silence) is True
        m.current_gpu_temp = lambda: 90.0
        assert m.wait_for_gpu_cool_down(80.0, 0.0, log=_silence) is True
        m.current_gpu_temp = lambda: None
        assert m.wait_for_gpu_cool_down(80.0, 8.0, log=_silence) is True
        # hot -> cools over a couple of fast polls -> True
        seq = iter([90.0, 80.0, 70.0])
        m.current_gpu_temp = lambda: next(seq)
        assert m.wait_for_gpu_cool_down(
            80.0, 8.0, poll_interval=0.01, max_wait=1.0, log=_silence
        ) is True
    finally:
        m.current_gpu_temp = orig


def main() -> int:
    suites = [
        ("python reference solutions", test_python_reference_solutions),
        ("javascript reference solutions", test_javascript_reference_solutions),
        ("bash reference solutions", test_bash_reference_solutions),
        ("tool reference plans", test_tool_reference_plans),
        ("wrong solutions rejected", test_wrong_solutions_are_rejected),
        ("thermal-throttle detection", test_thermal_throttle_detection),
        ("discover-models (all)", test_discover_models),
        ("gpu cool-down gate", test_gpu_cool_down),
    ]
    failures = []
    for name, fn in suites:
        try:
            fn()
        except AssertionError as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {exc}")
        else:
            print(f"  PASS  {name}")
    print()
    if failures:
        print("FAILED:", failures)
        return 1
    print("ALL TASK-DEFINITION TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
