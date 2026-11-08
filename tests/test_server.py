"""The local engine server: access checks, projects, streaming generation, live events and export.

A real server runs as a subprocess on a throwaway data folder, serving a tiny model that is
pre-trained in-process first.  Run from the repository root:  .venv/bin/python -m unittest tests.test_server -v
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from unittest import mock

from tests.support import isolated_environment, tiny_project

from model_builder import paths, worker

_server: subprocess.Popen | None = None
_info: dict = {}
_trained = None


def setUpModule() -> None:
    global _server, _info, _trained
    environment = isolated_environment("lmb-test-server-").start()
    unittest.addModuleCleanup(environment.stop)
    _trained = tiny_project("Served model", steps=4)
    saved = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        with mock.patch("os.nice"):
            if worker.run(_trained.id, "pretrain", "cpu", "test") != 0:
                raise RuntimeError((_trained.run_dir("pretrain") / "status.json").read_text())
    finally:
        for signum, handler in saved.items():
            signal.signal(signum, handler)

    child_env = {key: value for key, value in os.environ.items() if key != "LD_PRELOAD"}
    _server = subprocess.Popen([sys.executable, "-m", "model_builder", "serve", "--persist"], env=child_env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    unittest.addModuleCleanup(_stop_server)
    info_path = paths.runtime_dir() / "server.json"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if _server.poll() is not None:
            raise RuntimeError(f"The server exited early:\n{_server.stderr.read().decode(errors='replace')}")
        try:
            _info = json.loads(info_path.read_text())
            if request("GET", "/api/ping")[0].status == 200:
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise RuntimeError("The server did not start within a minute.")


def _stop_server() -> None:
    if _server and _server.poll() is None:
        _server.send_signal(signal.SIGTERM)
        try:
            _server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _server.kill()
            _server.wait()
    if _server and _server.stderr:
        _server.stderr.close()


def request(method: str, path: str, body=None, token: bool = True, headers: dict | None = None, host: str | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", _info["port"], timeout=120)
    sent = {"Host": host or f"127.0.0.1:{_info['port']}"}
    if token:
        sent["X-LMB-Token"] = _info["token"]
    payload = None
    if body is not None:
        payload = json.dumps(body).encode()
        sent["Content-Type"] = "application/json"
    sent.update(headers or {})
    try:
        connection.request(method, path, body=payload, headers=sent)
        response = connection.getresponse()
        return response, response.read()
    finally:
        connection.close()


class AccessTests(unittest.TestCase):
    def test_api_needs_the_session_token(self) -> None:
        self.assertEqual(request("GET", "/api/state", token=False)[0].status, 401)
        self.assertEqual(request("GET", "/api/state", token=False, headers={"X-LMB-Token": "wrong"})[0].status, 401)
        self.assertEqual(request("GET", "/?token=wrong", token=False)[0].status, 401)

    def test_token_link_sets_a_strict_http_only_cookie(self) -> None:
        response, _ = request("GET", f"/?token={_info['token']}", token=False)
        self.assertEqual((response.status, response.getheader("Location")), (302, "/"))
        cookie = response.getheader("Set-Cookie")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        response, _ = request("GET", "/api/ping", token=False, headers={"Cookie": cookie.split(";")[0]})
        self.assertEqual(response.status, 200)

    def test_foreign_hosts_and_cross_origin_writes_are_refused(self) -> None:
        self.assertEqual(request("GET", "/api/ping", host="attacker.example")[0].status, 403)
        response, _ = request("POST", "/api/settings", {}, headers={"Origin": "http://attacker.example"})
        self.assertEqual(response.status, 403)
        response, _ = request("POST", "/api/settings", {}, headers={"Origin": f"http://127.0.0.1:{_info['port']}"})
        self.assertEqual(response.status, 200)

    def test_page_has_a_content_security_policy_and_files_stay_inside_the_ui(self) -> None:
        response, body = request("GET", "/", token=False)
        self.assertEqual(response.status, 200)
        self.assertIn("script-src 'self'", response.getheader("Content-Security-Policy"))
        for path in ("/../model_builder/server.py", "/%2e%2e/model_builder/server.py", "/js/../../pyproject.toml"):
            response, body = request("GET", path, token=False)
            self.assertEqual(response.status, 404, path)


class ProjectApiTests(unittest.TestCase):
    def test_state_lists_projects_devices_and_settings(self) -> None:
        response, body = request("GET", "/api/state")
        state = json.loads(body)
        self.assertLessEqual({"app", "current", "devices", "projects", "settings", "job", "download"}, set(state))
        self.assertIn(_trained.id, [project["id"] for project in state["projects"]])
        self.assertIn("cpu", [device["id"] for device in state["devices"]])

    def test_invalid_changes_are_explained(self) -> None:
        response, body = request("POST", "/api/projects", {"name": "Scratch", "preset": "pocket"})
        self.assertEqual(response.status, 200, body)
        scratch = json.loads(body)["project"]["id"]
        try:
            response, body = request("PATCH", "/api/projects/current", {"config": {"context": 99_999}})
            self.assertEqual(json.loads(body)["config"]["context"], 1024)
            response, body = request("PATCH", "/api/projects/current", {"config": {"heads": 3}})
            self.assertEqual(response.status, 409)
            self.assertIn("heads", json.loads(body)["error"])
            response, body = request("PATCH", "/api/projects/current", {"data": {"pretrain": "no-such-dataset"}})
            self.assertEqual(response.status, 404)
            response, body = request("POST", "/api/projects/current/generate", {"prompt": "hi"})
            self.assertEqual(response.status, 409)  # nothing trained yet
        finally:
            self.assertEqual(request("DELETE", f"/api/projects/{scratch}")[0].status, 200)

    def test_trained_model_shape_is_locked(self) -> None:
        request("POST", "/api/projects/select", {"id": _trained.id})
        response, body = request("PATCH", "/api/projects/current", {"config": {"blocks": 2}})
        self.assertEqual(response.status, 409)
        self.assertIn("Reset pre-training", json.loads(body)["error"])


class ModelApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertEqual(request("POST", "/api/projects/select", {"id": _trained.id})[0].status, 200)

    def test_generation_streams_tokens_with_alternatives(self) -> None:
        response, body = request("POST", "/api/projects/current/generate",
                                 {"prompt": "The cat", "settings": {"max_tokens": 5, "seed": 3}})
        self.assertEqual(response.getheader("Content-Type").split(";")[0], "application/x-ndjson")
        lines = [json.loads(line) for line in body.splitlines() if line.strip()]
        self.assertEqual((lines[0]["type"], lines[-1]["type"]), ("start", "end"))
        tokens = [line["token"] for line in lines if line["type"] == "token"]
        self.assertTrue(1 <= len(tokens) <= 5)
        self.assertLessEqual({"id", "text", "prob", "alternatives"}, set(tokens[0]))
        again = request("POST", "/api/projects/current/generate", {"prompt": "The cat", "settings": {"max_tokens": 5, "seed": 3}})[1]
        self.assertEqual(again.splitlines()[1:-1], body.splitlines()[1:-1])  # same seed, same tokens

    def test_attention_maps_cover_every_block_and_head(self) -> None:
        response, body = request("POST", "/api/projects/current/attention", {"text": "the cat sat"})
        result = json.loads(body)
        self.assertEqual(response.status, 200, body)
        self.assertEqual(len(result["layers"]), result["blocks"])
        self.assertEqual(len(result["layers"][0]), result["heads"])
        count = len(result["tokens"])
        self.assertEqual(len(result["layers"][0][0]), count)

    def test_export_is_a_zip_that_loads_without_this_app(self) -> None:
        response, body = request("POST", "/api/projects/current/export", {"stage": "pretrain"})
        self.assertEqual(response.status, 200, body)
        path = json.loads(body)["path"]
        self.assertTrue(path.startswith(os.environ["XDG_DOWNLOAD_DIR"]))
        folder = tempfile.mkdtemp(prefix="lmb-export-")
        self.addCleanup(shutil.rmtree, folder, True)
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(set(archive.namelist()),
                             {"model.safetensors", "config.json", "tokenizer.json", "README.md", "model.py", "config.py"})
            archive.extractall(folder)
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertIn("<|assistant|>\n", config["chat_template"])
        script = (
            "import json, sys, torch\n"
            "from safetensors.torch import load_file\n"
            "from tokenizers import Tokenizer\n"
            "from config import ModelConfig\n"
            "from model import GPT\n"
            "model = GPT(ModelConfig.from_dict(json.load(open('config.json')))).eval()\n"
            "model.load_state_dict(load_file('model.safetensors'))\n"
            "ids = Tokenizer.from_file('tokenizer.json').encode('The cat').ids\n"
            "print(tuple(model(torch.tensor([ids])).shape), 'model_builder' in sys.modules)\n"
        )
        env = {key: value for key, value in os.environ.items() if key != "LD_PRELOAD"}
        result = subprocess.run([sys.executable, "-c", script], cwd=folder, env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip().endswith("False"), result.stdout)


class EventStreamTests(unittest.TestCase):
    def test_changes_are_pushed_to_listeners(self) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", _info["port"], timeout=30)
        self.addCleanup(connection.close)
        connection.request("GET", "/api/events", headers={"X-LMB-Token": _info["token"]})
        response = connection.getresponse()
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        self.assertEqual(response.fp.readline(), b"retry: 2000\n")
        request("POST", "/api/projects/select", {"id": _trained.id})
        while True:
            line = response.fp.readline()
            self.assertTrue(line, "the event stream closed")
            if line.startswith(b"data: ") and json.loads(line[6:])["type"] == "state":
                break

    def test_closing_the_window_without_a_job_changes_nothing(self) -> None:
        response, body = request("POST", "/api/window-closed", {})
        self.assertEqual(json.loads(body), {"stopped": False})


if __name__ == "__main__":
    unittest.main()
