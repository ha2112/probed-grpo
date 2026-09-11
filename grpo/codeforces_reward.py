"""Afterburner-shaped correctness and format reward for CodeContests."""

import argparse
import itertools
import json
import os
import re
from multiprocessing.pool import ThreadPool


MONOLITH_URL = os.environ.get("MONOLITH_URL", "https://monolith.cool/execute")
RESULT_PATTERN = re.compile(r"CODECONTESTS_RESULT:(\d+)/(\d+)")


def valid_format(text):
    pattern = re.compile(
        r"\A\s*<thinking>(?:(?!<thinking>).)*?</thinking>\s*"
        r"<solution>(?:(?!<thinking>|<solution>).)*?</solution>\s*\Z",
        re.DOTALL,
    )
    return bool(pattern.fullmatch(text))


def extract_code(text):
    solution_blocks = re.findall(r"<solution>(.*?)</solution>", text, flags=re.DOTALL)
    if not solution_blocks:
        return ""
    code_blocks = re.findall(r"```[\w+-]*(?:\n|\r\n)?(.*?)```", solution_blocks[-1], flags=re.DOTALL)
    return code_blocks[-1].strip() if code_blocks else ""


def _extract_stdout(payload):
    if not isinstance(payload, dict):
        return None
    containers = (payload.get("output_dict"), payload, payload.get("data"))
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("stdout", "output"):
            if isinstance(container.get(key), str):
                return container[key]
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("outputs"), list):
        parts = [
            item.get("data", "")
            for item in data["outputs"]
            if isinstance(item, dict)
            and item.get("type") == "stdout"
            and isinstance(item.get("data"), str)
        ]
        return "".join(parts) if parts else None
    return None


def _runner_code(solution_code, tests, per_case_timeout):
    return f'''import subprocess
import sys

SOLUTION = {solution_code!r}
TESTS = {tests!r}
TIMEOUT = {per_case_timeout!r}

def normalize(text):
    return text.strip().split()

passed = 0
for case in TESTS:
    try:
        result = subprocess.run(
            [sys.executable, "-c", SOLUTION],
            input=case["input"],
            text=True,
            capture_output=True,
            timeout=TIMEOUT,
        )
        if result.returncode == 0 and normalize(result.stdout) == normalize(case["output"]):
            passed += 1
    except Exception:
        pass

print(f"CODECONTESTS_RESULT:{{passed}}/{{len(TESTS)}}")
'''


def evaluate_solution(solution_code, tests, time_limit_seconds=0, *, raise_on_error=True):
    if not solution_code or len(tests) == 0:
        return False
    import requests

    per_case_timeout = max(1.0, min(10.0, 2.0 * float(time_limit_seconds or 1.0)))
    try:
        # Eight cases fit the service's 90-second budget even at the 10s limit.
        for start in range(0, len(tests), 8):
            cases = tests[start:start + 8]
            request = {
                "code": _runner_code(solution_code, cases, per_case_timeout),
                "language": "python", "libraries": [], "timeout": 90,
                "run_profiling": False,
            }
            response = requests.post(MONOLITH_URL, json=request, timeout=95)
            response.raise_for_status()
            stdout = _extract_stdout(response.json())
            match = RESULT_PATTERN.search(stdout or "")
            if not match:
                raise RuntimeError("sandbox response did not contain a CodeContests result")
            passed, total = map(int, match.groups())
            if total != len(cases) or not 0 <= passed <= total:
                raise RuntimeError("sandbox returned an inconsistent case count")
            if passed != total:
                return False
        return True
    except Exception as error:
        if raise_on_error:
            raise RuntimeError(f"CodeContests sandbox request failed: {error}") from error
        return False


def correctness_transition(baseline_passed, candidate_passed):
    if baseline_passed and candidate_passed:
        return 0.5
    if not baseline_passed and candidate_passed:
        return 1.0
    if baseline_passed and not candidate_passed:
        return -1.0
    return -0.5


def combine_reward(baseline_passed, candidate_passed, format_passed):
    # Afterburner disables length reward and weights improvement/format by 0.5/0.2.
    return 0.5 * correctness_transition(baseline_passed, candidate_passed) + 0.2 * float(format_passed)


def _decode_ground_truth(value):
    return json.loads(value) if isinstance(value, str) else value


def codeforces_reward_fn_batch(data_sources, solution_strs, ground_truths, extra_infos=None):
    ground_truths = [_decode_ground_truth(value) or {} for value in ground_truths]
    if extra_infos is None:
        extra_infos = list(itertools.repeat({}, len(solution_strs)))
    if not (len(data_sources) == len(solution_strs) == len(ground_truths) == len(extra_infos)):
        raise ValueError("Reward batch lengths differ")
    if len(solution_strs) == 0:
        return []
    jobs = [
        (
            extract_code(solution),
            ground_truth.get("tests", []),
            (extra_info or {}).get("time_limit_seconds", 0),
        )
        for solution, ground_truth, extra_info in zip(solution_strs, ground_truths, extra_infos)
    ]
    workers = int(os.environ.get("JUDGE_WORKERS", "8"))
    if workers < 1:
        raise ValueError("JUDGE_WORKERS must be positive")
    with ThreadPool(min(len(jobs), workers)) as pool:
        passed = pool.starmap(evaluate_solution, jobs)
    return [
        combine_reward(
            bool(ground_truth.get("baseline_passed", True)),
            candidate_passed,
            valid_format(solution),
        )
        for solution, ground_truth, candidate_passed in zip(solution_strs, ground_truths, passed)
    ]


def check_judge():
    tests = [{"input": "probe-ok\n", "output": "probe-ok\n"}]
    if not evaluate_solution("print(input())", tests):
        raise RuntimeError("CodeContests sandbox health check failed")
    if evaluate_solution("print('wrong-answer')", tests):
        raise RuntimeError("CodeContests sandbox accepted an incorrect answer")
    print(f"CodeContests reward sandbox is healthy: {MONOLITH_URL}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_judge()


if __name__ == "__main__":
    main()
