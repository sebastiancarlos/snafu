#!/usr/bin/env python
"""End-to-end test suite for snafu."""

import json
import os
import re
import subprocess
import tempfile
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


PYTHON_SOURCE = textwrap.dedent(
    """\
    def calculate_total(items):
        return items

    class ShoppingCart:
        def add_item(self):
            pass
    """
)

RUBY_SOURCE = textwrap.dedent(
    """\
    class ShoppingCart
      def calculate_total
      end
    end
    """
)

GO_SOURCE = textwrap.dedent(
    """\
    func CalculateTotal(x int) int { return x }
    func (u *UserStore) GetUserByID(id int) {}
    """
)

JAVA_SOURCE = textwrap.dedent(
    """\
    public class ShoppingCart {
        public void addItem(String s) {}
    }
    """
)

JAVASCRIPT_SOURCE = textwrap.dedent(
    """\
    function calculateTotal(items) { return items; }
    class ShoppingCart {
      calc() {}
    }
    """
)


SAMPLES = {
    "python": (".py", PYTHON_SOURCE, ["calculate_total", "ShoppingCart", "add_item"]),
    "ruby": (".rb", RUBY_SOURCE, ["ShoppingCart", "calculate_total"]),
    "go": (".go", GO_SOURCE, ["CalculateTotal", "GetUserByID"]),
    "java": (".java", JAVA_SOURCE, ["ShoppingCart", "addItem"]),
    "javascript": (".js", JAVASCRIPT_SOURCE, ["calculateTotal", "ShoppingCart"]),
}


def _cmd(*args: str) -> list[str]:
    # Prefer the project's own console script installed by `uv sync`.
    local = ROOT / ".venv" / "bin" / "snafu"
    if local.is_file():
        return [str(local), *args]
    return ["uv", "run", "snafu", *args]


def snafu(
    *args: str,
    cwd: str | Path = ROOT,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        _cmd(*args),
        cwd=cwd,
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=60,
    )


class FakeLLMHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/v1/models"):
            self._send(200, {"object": "list", "data": [{"id": "gpt-fake"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        content = req["messages"][-1]["content"]
        if '"new_symbol_makes_sense"' in content:
            payload = {
                "new_symbol_makes_sense": True,
                "first_alternative_makes_sense": True,
                "reason": "fake server says the rename is fine",
            }
        elif '"new_symbol"' in content:
            payload = {"new_symbol": "get_user_by_id"}
        elif '"interpretations"' in content:
            if "Symbol name: `get_user_by_id`" in content:
                payload = {
                    "interpretations": [{"description": "fetch a user by id", "probability": 1.0}]
                }
            else:
                payload = {
                    "interpretations": [
                        {"description": "perform an action on some entity", "probability": 0.6},
                        {"description": "compute a derived value", "probability": 0.4},
                    ]
                }
        else:
            self._send(400, {"error": "no matching prompt"})
            return
        self._send(
            200,
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-fake",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(payload)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


class FakeLLMServer:
    def __init__(self) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLMHandler)
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    @property
    def env(self) -> dict[str, str]:
        e = dict(os.environ)
        e["OPENAI_API_KEY"] = "sk-test-fake"
        e["OPENAI_BASE_URL"] = f"http://127.0.0.1:{self.port}/v1"
        return e

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


class TestCLI(unittest.TestCase):
    def test_help_exits_zero(self) -> None:
        res = snafu("--help")
        self.assertEqual(res.returncode, 0)
        self.assertIn("--model", res.stdout)
        self.assertIn("OPENAI_API_KEY", res.stdout)

    def test_unsupported_extension_fails(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "contract.sol"
            path.write_text("contract Foo {}\n")
            res = snafu(str(path))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Unsupported file type", res.stderr)

    def test_bad_python_syntax_fails(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "broken.py"
            path.write_text("def broken(:\n")
            res = snafu(str(path))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Could not parse", res.stderr)

    def test_no_multiword_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "simple.py"
            path.write_text("def foo():\n    return 1\n")
            res = snafu(str(path))
        self.assertEqual(res.returncode, 0)
        self.assertIn("No multi-word symbols found.", res.stdout)

    def test_dry_run_lists_symbols_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sample.py"
            path.write_text("def calculate_total(items):\n    return items\n")
            res = snafu(str(path), "--dry-run")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("calculate_total", strip_ansi(res.stdout))
        self.assertNotIn("Round 1", res.stderr)

    def test_symbols_file_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            symfile = Path(d) / "symbols.txt"
            symfile.write_text("fetch_user_data\n\n  get_user_by_id  \n")
            res = snafu("--symbols-file", str(symfile), "--dry-run")
        self.assertEqual(res.returncode, 0, res.stderr)
        out = strip_ansi(res.stdout)
        self.assertIn("fetch_user_data", out)
        self.assertIn("get_user_by_id", out)

    def test_no_source_errors(self) -> None:
        res = snafu()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("provide either a source file or --symbols-file", res.stderr)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = FakeLLMServer()
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()
        cls.server.close()

    def test_pipeline_success_for_each_language(self) -> None:
        for name, (ext, source, expected) in SAMPLES.items():
            with self.subTest(language=name):
                path = Path(self.tmp.name) / f"sample{ext}"
                path.write_text(source)
                res = snafu(
                    str(path),
                    "--model",
                    "openai:gpt-fake",
                    env=self.server.env,
                    input_text="1\n" * 30,
                )
                self.assertEqual(res.returncode, 0, res.stderr)
                out = strip_ansi(res.stdout)
                self.assertIn("RESULTS", out)
                self.assertIn("SUCCESS", out)
                self.assertIn("-> get_user_by_id", out)
                self.assertIn("fully validated as success:", out)
                for sym in expected:
                    self.assertIn(sym, out)

    def test_pipeline_with_symbols_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            symfile = Path(d) / "symbols.txt"
            symfile.write_text("calculate_total\n")
            res = snafu(
                "--symbols-file",
                str(symfile),
                "--model",
                "openai:gpt-fake",
                env=self.server.env,
                input_text="1\n" * 30,
            )
        self.assertEqual(res.returncode, 0, res.stderr)
        out = strip_ansi(res.stdout)
        self.assertIn("RESULTS", out)
        self.assertIn("SUCCESS", out)
        self.assertIn("-> get_user_by_id", out)
        self.assertIn("fully validated as success:", out)


if __name__ == "__main__":
    unittest.main()
