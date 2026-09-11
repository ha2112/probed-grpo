"""Loopback-only Monolith-compatible Python judge, isolated with Docker per request."""

import argparse
import json
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


JUDGE_IMAGE = "python@sha256:a8e0a3090316aed0b11037aac613aef32fb1747dcc1dcb5c0f6c727a0113a07f"
MAX_BODY = 16 * 1024 * 1024


def docker_command(name):
    return [
        "docker", "run", "--rm", "--interactive", "--name", name,
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", "64",
        "--memory", "1g", "--memory-swap", "1g", "--cpus", "2",
        "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--workdir", "/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--log-driver", "none", JUDGE_IMAGE, "python", "-I", "-",
    ]


def execute(payload):
    if payload.get("language") != "python" or payload.get("libraries"):
        raise ValueError("The bundled judge supports Python standard-library programs only")
    code = payload.get("code")
    timeout = payload.get("timeout", 90)
    if not isinstance(code, str) or not code or not isinstance(timeout, (int, float)) or not 0 < timeout <= 90:
        raise ValueError("Expected nonempty code and a timeout in (0, 90]")
    name = "probed-grpo-judge-" + uuid.uuid4().hex
    try:
        result = subprocess.run(docker_command(name), input=code, text=True,
                                capture_output=True, timeout=timeout + 3)
        if result.returncode != 0:
            raise RuntimeError(f"Isolated runner failed ({result.returncode}): {result.stderr[-2000:]}")
        if len(result.stdout) > 1024 * 1024:
            raise RuntimeError("Runner output exceeded 1 MiB")
        return {"stdout": result.stdout}
    finally:
        # Also removes the exact request container after a timeout/interruption.
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=10)


class JudgeHandler(BaseHTTPRequestHandler):
    def respond(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/execute":
            self.respond(404, {"error": "Use POST /execute"})
            return
        acquired = False
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("Request body must be between 1 byte and 16 MiB")
            self.connection.settimeout(10)
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            acquired = self.server.slots.acquire(timeout=1)
            if not acquired:
                self.respond(503, {"error": "Judge busy; reduce JUDGE_WORKERS"})
                return
            self.respond(200, execute(payload))
        except (ValueError, TypeError) as error:
            self.respond(400, {"error": str(error)})
        except Exception as error:
            self.respond(500, {"error": str(error)})
        finally:
            if acquired:
                self.server.slots.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-image", action="store_true", help="Download the pinned judge image and exit")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.pull_image:
        subprocess.run(["docker", "pull", JUDGE_IMAGE], check=True)
        return
    subprocess.run(["docker", "image", "inspect", JUDGE_IMAGE], check=True, stdout=subprocess.DEVNULL)
    # Refuse to serve until Docker isolation actually starts and executes Python.
    result = execute({"language": "python", "code": "print('judge-ready')", "timeout": 10})
    if result["stdout"].strip() != "judge-ready":
        raise SystemExit("Judge self-test failed")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), JudgeHandler)
    server.slots = threading.BoundedSemaphore(args.workers)
    print(f"Judge ready at http://127.0.0.1:{args.port}/execute ({args.workers} workers)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
