"""Correctness-only Venus reward: pass all tests + format (no time/memory).

Judge backends (env ``VENUS_JUDGE``):
  - ``local``  — run harness with local Python subprocess (default on clusters
    that cannot reach Monolith)
  - ``monolith`` — POST to ``MONOLITH_URL``
  - ``auto`` — try Monolith once; on network/DNS failure fall back to local
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from multiprocessing.pool import ThreadPool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_cache import VENUS_DATASET_ID  # noqa: E402
from grpo.venus_reward import MONOLITH_URL, runner_code  # noqa: E402

_JUDGE_MODE = None
_JUDGE_WARNED = False
_FENCE = re.compile(r"```[\w+-]*(?:\n|\r\n)?(.*?)```", re.DOTALL)
_SOLUTION = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)


_TAIL_CYCLE = re.compile(r"(.{1,8}?)\1{7,}\s*$", re.DOTALL)


def _strip_tail_cycle(text: str) -> str:
    """Drop a trailing token loop (`1, 1, 1` / `000` / `(((`) without touching the code before it."""
    match = _TAIL_CYCLE.search(text or "")
    if not match:
        return text or ""
    return text[: match.start()].rstrip()


def longest_compilable_prefix(code: str) -> str:
    """Longest leading line-prefix that compiles. Cheap: stops at the first success from the end."""
    raw = (code or "").strip()
    if not raw:
        return ""
    if code_is_valid(raw):
        return raw
    lines = raw.splitlines()
    for end in range(len(lines) - 1, 0, -1):
        chunk = "\n".join(lines[:end]).strip()
        if chunk and code_is_valid(chunk):
            return chunk
    return ""


def _code_from_body(body: str) -> str:
    """First code fence, else the unclosed body. Then drop a trailing loop and keep a compilable prefix."""
    if not body:
        return ""
    fenced = _FENCE.findall(body)
    chunk = ""
    for piece in fenced:
        if piece.strip():
            chunk = piece.strip()
            break
    if not chunk:
        opened = re.search(r"```[\w+-]*(?:\n|\r\n)?(.*)\Z", body, re.DOTALL)
        chunk = opened.group(1).split("```", 1)[0] if opened else body
        chunk = re.split(r"</solution>", chunk, maxsplit=1, flags=re.IGNORECASE)[0]
    chunk = _strip_tail_cycle(chunk).strip()
    compiled = longest_compilable_prefix(chunk)
    return compiled or chunk


def extract_solve_code(text: str) -> str:
    """Extract the Python the model actually wrote, even if it never closed </solution>."""
    if not text:
        return ""
    blocks = _SOLUTION.findall(text)
    if blocks:
        code = _code_from_body(blocks[-1])
        if code:
            return code
    if "<solution>" in text.lower():
        tail = re.split(r"<solution>", text, maxsplit=1, flags=re.IGNORECASE)[-1]
        code = _code_from_body(tail)
        if code:
            return code
    opened = re.search(r"```[\w+-]*(?:\n|\r\n)?(.*)\Z", text, re.DOTALL)
    body = opened.group(1).split("```", 1)[0] if opened else text
    fenced = _FENCE.findall(text)
    for piece in fenced:
        if piece.strip():
            body = piece
            break
    return _code_from_body(body)


def gradient_continuation(continuation: str) -> str:
    """Prefix of newly generated text that GRPO is allowed to reinforce.

    Includes the closing fence when the model emitted one. Drops text after a
    token loop so a `1, 1, 1` tail cannot dominate the advantage.
    """
    text = continuation or ""
    close = text.find("```")
    tag = text.lower().find("</solution>")
    ends = [index for index in (close, tag) if index >= 0]
    if ends:
        index = min(ends)
        if close >= 0 and index == close:
            return text[: close + 3]
        return text[: tag + len("</solution>")]
    trimmed = _strip_tail_cycle(text)
    kept = longest_compilable_prefix(trimmed) or trimmed.strip()
    return kept


def format_score(text: str) -> float:
    """Closed compilable code outranks an unclosed compilable body. Tags alone score 0."""
    code = extract_solve_code(text)
    syntax = code_is_valid(code)
    closed = "</solution>" in text.lower() and text.count("```") >= 2
    if closed and syntax:
        return 1.0
    if syntax:
        return 0.6
    return 0.0


_SCORE = re.compile(r"^VENUS_SCORE:(\d+)/(\d+)$")


def code_is_valid(code: str) -> bool:
    """True when extracted Python compiles. Does not execute it."""
    if not code or not code.strip():
        return False
    try:
        compile(code, "<candidate>", "exec")
    except (SyntaxError, ValueError, TypeError):
        return False
    return True


def _parse_harness_stdout(stdout: str) -> bool:
    detail = _parse_harness_detail(stdout)
    if detail["verdict"] == "success":
        return True
    if detail["verdict"] == "failed":
        return False
    raise ValueError(f"Venus harness returned unexpected stdout: {(stdout or '')[:200]!r}")


def _parse_harness_detail(stdout: str) -> dict:
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    tests_passed = None
    tests_total = None
    for line in lines:
        match = _SCORE.match(line)
        if match:
            tests_passed = int(match.group(1))
            tests_total = int(match.group(2))
    if last == "Success":
        verdict = "success"
    elif last == "Failed":
        verdict = "failed"
    else:
        verdict = "unknown"
    return {
        "verdict": verdict,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
    }


def _n_tests(instance, case_multiply=1) -> int:
    tests = instance.get("test_cases")
    if isinstance(tests, str):
        tests = json.loads(tests)
    return int(len(tests or []) * int(case_multiply))


def _run_harness(solution, instance, case_multiply, timeout=90):
    program = runner_code(solution, instance, case_multiply)
    with tempfile.TemporaryDirectory(prefix="venus_judge_") as directory:
        path = Path(directory) / "harness.py"
        path.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=directory,
                env={**os.environ, "PYTHONPATH": ""},
            )
        except subprocess.TimeoutExpired:
            return None
    stream = completed.stdout or ""
    if "Success" not in stream and "Failed" not in stream:
        stream = (completed.stdout or "") + "\n" + (completed.stderr or "")
    return stream


def evaluate_local(solution, instance, case_multiply, timeout=90):
    """Execute Venus unittest harness in a local Python subprocess."""
    stream = _run_harness(solution, instance, case_multiply, timeout=timeout)
    if stream is None:
        return False
    try:
        return _parse_harness_stdout(stream)
    except ValueError:
        return False


def evaluate_local_detail(solution, instance, case_multiply, timeout=90) -> dict:
    """Pass-all plus syntax and per-test fraction. Does not change GRPO reward."""
    total = _n_tests(instance, case_multiply)
    syntax_ok = code_is_valid(solution)
    if not solution or not solution.strip():
        return {
            "passed": False,
            "syntax_ok": False,
            "tests_passed": 0,
            "tests_total": total,
            "test_fraction": 0.0,
        }
    stream = _run_harness(solution, instance, case_multiply, timeout=timeout)
    if stream is None:
        return {
            "passed": False,
            "syntax_ok": syntax_ok,
            "tests_passed": 0,
            "tests_total": total,
            "test_fraction": 0.0,
        }
    detail = _parse_harness_detail(stream)
    passed = detail["verdict"] == "success"
    tests_passed = detail["tests_passed"]
    tests_total = detail["tests_total"] if detail["tests_total"] else total
    if tests_passed is None:
        tests_passed = tests_total if passed else 0
    if tests_total is None or tests_total < 1:
        tests_total = total
    fraction = 0.0 if tests_total < 1 else float(tests_passed) / float(tests_total)
    return {
        "passed": passed,
        "syntax_ok": syntax_ok,
        "tests_passed": int(tests_passed),
        "tests_total": int(tests_total),
        "test_fraction": fraction,
    }


def evaluate_monolith(solution, instance, case_multiply, timeout=90):
    import requests

    request = {
        "code": runner_code(solution, instance, case_multiply),
        "language": "python",
        "libraries": [],
        "timeout": timeout,
        "run_profiling": False,
    }
    response = requests.post(MONOLITH_URL, json=request, timeout=timeout + 5)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "timeout":
        return False
    if payload.get("status") != "success":
        raise ValueError(f"Venus judge status={payload.get('status')!r}")
    stdout = ""
    if isinstance(payload.get("output_dict"), dict):
        stdout = str(payload["output_dict"].get("stdout", "")).strip()
    if not stdout:
        from grpo.codeforces_reward import _extract_stdout

        stdout = (_extract_stdout(payload) or "").strip()
    return _parse_harness_stdout(stdout)


def resolve_judge_mode():
    global _JUDGE_MODE, _JUDGE_WARNED
    configured = os.environ.get("VENUS_JUDGE", "auto").strip().lower()
    if configured in {"local", "monolith"}:
        return configured
    if configured != "auto":
        raise ValueError(f"Unknown VENUS_JUDGE={configured!r}; use local|monolith|auto")
    if _JUDGE_MODE is not None:
        return _JUDGE_MODE
    try:
        import socket
        from urllib.parse import urlparse

        host = urlparse(MONOLITH_URL).hostname or "monolith.cool"
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        _JUDGE_MODE = "monolith"
    except OSError:
        _JUDGE_MODE = "local"
        if not _JUDGE_WARNED:
            print(
                f"VENUS_JUDGE=auto: cannot resolve Monolith ({MONOLITH_URL}); "
                "using local Python harness",
                flush=True,
            )
            _JUDGE_WARNED = True
    return _JUDGE_MODE


def evaluate_solution_correctness(solution, extra_info, *, raise_on_error=True):
    """Run Venus harness once; return whether all tests passed (no profiling)."""
    if not solution:
        return False
    instance = extra_info["instance"]
    if isinstance(instance, str):
        instance = json.loads(instance)
    case_multiply = int(extra_info.get("case_multiply", 1))
    mode = resolve_judge_mode()
    try:
        if mode == "local":
            return evaluate_local(solution, instance, case_multiply)
        try:
            return evaluate_monolith(solution, instance, case_multiply)
        except Exception as error:
            if os.environ.get("VENUS_JUDGE", "auto").strip().lower() == "auto":
                global _JUDGE_MODE, _JUDGE_WARNED
                _JUDGE_MODE = "local"
                if not _JUDGE_WARNED:
                    print(
                        f"Monolith judge failed ({error}); falling back to local harness",
                        flush=True,
                    )
                    _JUDGE_WARNED = True
                return evaluate_local(solution, instance, case_multiply)
            raise
    except Exception as error:
        if raise_on_error:
            raise RuntimeError(f"Venus correctness judge failed: {error}") from error
        return False


def combine_solve_reward(passed: bool, format_value: float, test_fraction: float = 0.0) -> float:
    """A full pass is always 1. Partial tests + syntax can break ties but cannot beat a pass."""
    if passed:
        return 1.0
    shaped = 0.25 * float(test_fraction) + 0.20 * float(format_value)
    return min(0.45, shaped)


def _reward_one(job):
    code, info = job
    instance = info["instance"]
    if isinstance(instance, str):
        instance = json.loads(instance)
    case_multiply = int(info.get("case_multiply", 1))
    if not code:
        return False, 0.0
    try:
        detail = evaluate_local_detail(code, instance, case_multiply)
    except Exception:
        return False, 0.0
    return bool(detail["passed"]), float(detail["test_fraction"])


def venus_solve_reward_batch(solution_strs, extra_infos, *, raise_on_error=True):
    if len(solution_strs) != len(extra_infos):
        raise ValueError("solution_strs and extra_infos length mismatch")
    if len(solution_strs) == 0:
        return [], []
    workers = int(os.environ.get("JUDGE_WORKERS", "8"))
    if workers < 1:
        raise ValueError("JUDGE_WORKERS must be positive")
    resolve_judge_mode()
    if resolve_judge_mode() != "local":
        raise RuntimeError("Venus solve GRPO reward requires VENUS_JUDGE=local")
    codes = [extract_solve_code(text) for text in solution_strs]
    jobs = list(zip(codes, extra_infos))
    with ThreadPool(min(len(jobs), workers)) as pool:
        judged = list(pool.map(_reward_one, jobs))
    passed = [ok for ok, _ in judged]
    rewards = [
        combine_solve_reward(ok, format_score(text), fraction)
        for (ok, fraction), text in zip(judged, solution_strs)
    ]
    return rewards, passed


def _score_one(job):
    code, info = job
    instance = info["instance"]
    if isinstance(instance, str):
        instance = json.loads(instance)
    case_multiply = int(info.get("case_multiply", 1))
    return evaluate_local_detail(code, instance, case_multiply)


def venus_solve_score_batch(solution_strs, extra_infos):
    """Eval-only details: pass-all, syntax, and tests_passed/tests_total."""
    if len(solution_strs) != len(extra_infos):
        raise ValueError("solution_strs and extra_infos length mismatch")
    if len(solution_strs) == 0:
        return []
    workers = int(os.environ.get("JUDGE_WORKERS", "8"))
    codes = [extract_solve_code(text) for text in solution_strs]
    jobs = list(zip(codes, extra_infos))
    with ThreadPool(min(len(jobs), workers)) as pool:
        return list(pool.map(_score_one, jobs))


def venus_solve_reward_fn_batch(data_sources, solution_strs, ground_truths, extra_infos=None):
    """verl-compatible signature; ignores ground_truth (no baseline needed)."""
    if extra_infos is None:
        raise ValueError("venus_solve_reward requires extra_infos with Venus instances")
    if any(source != VENUS_DATASET_ID for source in data_sources):
        raise ValueError("venus_solve_reward received a non-Venus data_source")
    rewards, _ = venus_solve_reward_batch(solution_strs, extra_infos)
    return rewards


def check_judge():
    os.environ["VENUS_JUDGE"] = "local"
    global _JUDGE_MODE
    _JUDGE_MODE = "local"
    instance = {
        "test_case_runners": "==Code Submission==\nprint(Solution().echo(int(input())))",
        "test_case_evaluator": "def evaluate(expected, actual):\n    return int(expected) == int(actual)",
        "test_cases": json.dumps([{"input": "7", "output": "7"}]),
    }
    info = {"instance": instance, "case_multiply": 1}
    code = "class Solution:\n    def echo(self, n):\n        return n"
    if not evaluate_solution_correctness(code, info):
        raise RuntimeError("Venus local judge rejected a correct solution")
    if evaluate_solution_correctness(code.replace("return n", "return 0"), info):
        raise RuntimeError("Venus local judge accepted an incorrect solution")
    wrapped = "Here is code:\n```python\n" + code + "\n```\n"
    if extract_solve_code(wrapped).strip() != code.strip():
        raise RuntimeError("extract_solve_code failed on markdown-only output")
    looped = "<solution>\n```python\n" + code + "\n" + ", ".join(["1"] * 24)
    recovered = extract_solve_code(looped)
    if "return n" not in recovered or recovered.count(", 1") > 4:
        raise RuntimeError(f"extract did not drop the token loop: {recovered[-80:]!r}")
    print("Venus correctness-only LOCAL judge is healthy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    if parser.parse_args().check:
        check_judge()
