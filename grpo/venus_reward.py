"""Afterburner correctness, efficiency and format rewards for Venus Python."""

import argparse
import json
import math
import os
import sys
import textwrap
from multiprocessing.pool import ThreadPool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from artifact_cache import VENUS_DATASET_ID  # noqa: E402
from grpo.codeforces_reward import correctness_transition, extract_code, valid_format  # noqa: E402

MONOLITH_URL = os.environ.get("MONOLITH_URL", "https://monolith.cool/execute")
CLIPS = {"time": 90, "memory": 1048576, "integral": 1048576 * 90}
FAILED = {"passed": False, "time": 90, "memory": 1e9, "integral": 1e10}

# Same execution scope and standard-library prelude as Afterburner's evaluator.
PRELUDE = """import io, re, itertools, collections, heapq, bisect, string, sys, functools, math, copy, unittest
from math import floor, ceil, factorial, sqrt, inf
from sys import maxsize, stdin
from bisect import bisect_left, bisect_right
from itertools import permutations, zip_longest
from heapq import heappush, heappop, heapify
from collections import deque, defaultdict, OrderedDict
from typing import List, Optional, Tuple
from functools import lru_cache, cache
"""


def runner_code(solution, instance, case_multiply):
    """Execute Venus's class/function runner and its problem-specific comparator."""
    if not isinstance(case_multiply, int) or case_multiply < 1:
        raise ValueError("case_multiply must be a positive integer")
    if "==Code Submission==" not in instance["test_case_runners"]:
        raise ValueError("Venus runner is missing its code placeholder")
    tests = json.loads(instance["test_cases"])
    if not tests:
        raise ValueError("Venus requires nonempty tests")
    program = instance["test_case_runners"].replace("==Code Submission==", solution.strip())
    # Compile inside the guarded invocation so candidate syntax errors fail tests.
    wrapped = "def running_solution():\n" + textwrap.indent(program.strip(), "    ")
    return PRELUDE + f'''
try:
    exec(compile({wrapped!r}, "<candidate>", "exec"))
except (SyntaxError, ValueError):
    def running_solution():
        raise RuntimeError("Invalid candidate code")

class TestSolution(unittest.TestCase):
    def run_io_fun(self, input_data):
        original_stdin, original_stdout = sys.stdin, sys.stdout
        try:
            sys.stdin = io.StringIO(input_data)
            captured = io.StringIO()
            sys.stdout = captured
            running_solution()
            return captured.getvalue()
        finally:
            sys.stdin, sys.stdout = original_stdin, original_stdout

def make_test_function(input_data, expected):
{textwrap.indent(instance["test_case_evaluator"].strip(), "    ")}
    def test_method(self):
        actual = self.run_io_fun(input_data)
        self.assertTrue(evaluate(expected, actual))
    return test_method

TESTS = {tests!r}
for i, case in enumerate(TESTS * {case_multiply}, start=1):
    setattr(TestSolution, f"test_case_{{i}}", make_test_function(case["input"], case["output"]))

if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    n_total = int(result.result.testsRun)
    n_fail = len(result.result.failures) + len(result.result.errors)
    n_pass = max(0, n_total - n_fail)
    print(f"VENUS_SCORE:{{n_pass}}/{{n_total}}")
    print("Success" if result.result.wasSuccessful() else "Failed")
'''


def evaluate_solution(solution, extra_info):
    if not solution:
        return dict(FAILED)
    import requests

    instance = extra_info["instance"]
    if isinstance(instance, str):
        instance = json.loads(instance)
    request = {"code": runner_code(solution, instance, extra_info["case_multiply"]),
               "language": "python", "libraries": [], "timeout": 90, "run_profiling": True}
    try:
        response = requests.post(MONOLITH_URL, json=request, timeout=95)
        response.raise_for_status()
        payload = response.json()
        # Only explicit execution timeouts are candidate failures; service errors abort.
        if payload.get("status") == "timeout":
            return dict(FAILED)
        output = payload.get("output_dict")
        if payload.get("status") != "success" or not isinstance(output, dict):
            raise ValueError("Venus needs a profiling Monolith judge (status=success and output_dict); "
                             "the bundled correctness-only judge is insufficient")
        measurements = {"time": output.get("duration"), "memory": output.get("peak_memory"),
                        "integral": output.get("integral")}
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in measurements.values()):
            raise ValueError("Missing or invalid Monolith profiling measurements")
        stdout = output.get("stdout", "").strip()
        if stdout not in {"Success", "Failed"}:
            raise ValueError("Venus sandbox did not return a harness verdict")
        return {"passed": stdout == "Success", **measurements} if stdout == "Success" else dict(FAILED)
    except Exception as error:
        raise RuntimeError(f"Venus sandbox request failed: {error}") from error


def combine_reward(baseline, current, objective, format_passed):
    clip = CLIPS[objective]
    before = min(clip, max(0, baseline[objective]))
    after = min(clip, max(0, current[objective]))
    gain = min(1.0, max(-1.0, (before - after) / (before + 1e-9)))
    transition = correctness_transition(baseline["passed"], current["passed"])
    improvement = transition + (0.5 * math.tanh(gain) if current["passed"] else 0)
    return 0.5 * improvement + 0.2 * float(format_passed)


def venus_reward_fn_batch(data_sources, solution_strs, ground_truths, extra_infos=None):
    if extra_infos is None or not (len(data_sources) == len(solution_strs) == len(ground_truths) == len(extra_infos)):
        raise ValueError("Venus reward requires equally sized data, response, baseline and metadata batches")
    if any(source != VENUS_DATASET_ID for source in data_sources):
        raise ValueError("Venus reward received a different dataset; rebuild the GRPO corpora")
    if len(solution_strs) == 0:
        return []
    baselines = [json.loads(v) if isinstance(v, str) else v for v in ground_truths]
    workers = int(os.environ.get("JUDGE_WORKERS", "8"))
    if workers < 1:
        raise ValueError("JUDGE_WORKERS must be positive")
    jobs = [(extract_code(s), info) for s, info in zip(solution_strs, extra_infos)]
    with ThreadPool(min(len(jobs), workers)) as pool:
        results = pool.starmap(evaluate_solution, jobs)
    return [combine_reward(b, r, info["efficiency_instruction"], valid_format(s))
            for b, r, info, s in zip(baselines, results, extra_infos, solution_strs)]


def check_judge():
    instance = {"test_case_runners": "==Code Submission==\nprint(Solution().echo(int(input())))",
                "test_case_evaluator": "def evaluate(expected, actual):\n    return int(expected) == int(actual)",
                "test_cases": json.dumps([{"input": "7", "output": "7"}])}
    info = {"instance": instance, "case_multiply": 64}
    code = "class Solution:\n    def echo(self, n):\n        return n"
    if not evaluate_solution(code, info)["passed"]:
        raise RuntimeError("Venus judge rejected the correct solution")
    if evaluate_solution(code.replace("return n", "return 0"), info)["passed"]:
        raise RuntimeError("Venus judge accepted an incorrect solution")
    print(f"Venus correctness and profiling judge is healthy: {MONOLITH_URL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    if parser.parse_args().check:
        check_judge()
