"""Platform differences: macOS folders, memory, GPU choice and windows; WSL folders, openers and Windows
browsers; and the checks that do without /proc.  They run anywhere by imitating the other systems.

Run from the repository root:  .venv/bin/python -m unittest tests.test_platform -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from model_builder import app, devices, paths, server, system
from model_builder import project as projects


def start(test: unittest.TestCase, *patches) -> None:
    for patch in patches:
        patch.start()
        test.addCleanup(patch.stop)


class MacTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = Path(tempfile.mkdtemp(prefix="lmb-mac-"))
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        start(self, mock.patch.object(system, "is_macos", return_value=True), mock.patch.object(system, "is_wsl", return_value=False),
              mock.patch.object(Path, "home", return_value=self.home), mock.patch.dict(os.environ))
        for variable in ("LMB_DATA_DIR", "LMB_CACHE_DIR", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
            os.environ.pop(variable, None)

    def test_folders_follow_the_mac_conventions(self) -> None:
        self.assertEqual(paths.data_dir(), self.home / "Library/Application Support/language-model-builder")
        self.assertEqual(paths.cache_dir(), self.home / "Library/Caches/language-model-builder")

    def test_memory_comes_from_sysctl(self) -> None:
        values = {"hw.memsize": str(16 << 30), "kern.memorystatus_level": "25"}
        with mock.patch.object(system, "sysctl", side_effect=lambda name: values.get(name, "")):
            self.assertEqual(projects._meminfo("MemTotal"), 16 << 30)
            self.assertEqual(projects._meminfo("MemAvailable"), 4 << 30)

    def test_auto_uses_the_metal_gpu_on_apple_silicon_only(self) -> None:
        with mock.patch.object(devices, "available_ids", return_value=["cpu", "mps"]):
            with mock.patch("platform.machine", return_value="arm64"):
                self.assertEqual(devices.resolve("auto"), "mps")
            with mock.patch("platform.machine", return_value="x86_64"):  # an Intel Mac's GPU only once timed faster
                self.assertEqual(devices.resolve("auto"), "cpu")
                self.assertEqual(devices.resolve("auto", {"calibrated": True, "auto_device": "mps"}), "mps")

    def test_chrome_in_applications_opens_without_linux_flags(self) -> None:
        chrome = self.home / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        chrome.parent.mkdir(parents=True)
        chrome.touch()
        self.assertIn(str(chrome), app.chromium_candidates())
        command = app._chromium_command(str(chrome), "http://127.0.0.1:1/")
        self.assertFalse([flag for flag in command if flag.startswith("--class")])

    def test_links_and_folders_open_with_open(self) -> None:
        [command] = system.opener_commands("/Users/me/Downloads")
        self.assertEqual((Path(command[0]).name, command[1]), ("open", "/Users/me/Downloads"))


class WslTests(unittest.TestCase):
    def setUp(self) -> None:
        start(self, mock.patch.object(system, "is_macos", return_value=False), mock.patch.object(system, "is_wsl", return_value=True))

    def test_downloads_go_to_the_windows_downloads_folder(self) -> None:
        folder = Path(tempfile.mkdtemp(prefix="lmb-wsl-")) / "Downloads"
        self.addCleanup(shutil.rmtree, folder.parent, ignore_errors=True)
        with mock.patch.dict(os.environ), mock.patch.object(system, "windows_downloads", return_value=folder):
            os.environ.pop("XDG_DOWNLOAD_DIR", None)
            self.assertEqual(paths.downloads_dir(), folder)

    def test_windows_variables_survive_non_ascii_user_names(self) -> None:
        system.windows_env.cache_clear()
        self.addCleanup(system.windows_env.cache_clear)
        replies = {"USERPROFILE": "C:\\Users\\Zoë\r\n", "NOPE": "%NOPE%\r\n"}

        def cmd(command, **_):  # cmd.exe /u answers in UTF-16
            return subprocess.CompletedProcess(command, 0, replies[command[-1].strip("%")].encode("utf-16-le"), b"")

        with mock.patch.object(system, "_windows_program", return_value="/mnt/c/Windows/System32/cmd.exe"), \
                mock.patch.object(system, "windows_drive", return_value=None), mock.patch("subprocess.run", side_effect=cmd):
            self.assertEqual(system.windows_env("USERPROFILE"), "C:\\Users\\Zoë")
            self.assertIsNone(system.windows_env("NOPE"))

    def test_folders_and_links_open_on_the_windows_side(self) -> None:
        explorer, share = "/mnt/c/Windows/explorer.exe", "\\\\wsl.localhost\\Ubuntu\\home\\me"
        with mock.patch("shutil.which", side_effect={"explorer.exe": explorer}.get), \
                mock.patch.object(system, "windows_path", return_value=share):
            self.assertEqual(system.opener_commands("/home/me"), [[explorer, share]])
            self.assertEqual(system.opener_commands("https://example.org/"), [[explorer, "https://example.org/"]])

    def test_windows_browsers_get_a_windows_profile_folder(self) -> None:
        edge = "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
        profile = "C:\\Users\\me\\AppData\\Local\\language-model-builder\\browser-profile"
        with mock.patch.object(system, "windows_browser_profile", return_value=profile):
            command = app._chromium_command(edge, "http://127.0.0.1:1/")
        self.assertIn(f"--user-data-dir={profile}", command)
        self.assertNotIn("--class=language-model-builder", command)
        with mock.patch.object(system, "windows_browser_profile", return_value=None):
            self.assertIsNone(app._chromium_command(edge, "http://127.0.0.1:1/"))


class BrowserWindowTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("true") and shutil.which("false") and shutil.which("sleep"), "needs true, false and sleep")
    def test_quick_exits_are_told_apart_from_closed_windows(self) -> None:
        with mock.patch.object(app, "QUICK_EXIT_SECONDS", 1.5):
            self.assertIsNone(app._wait_for_window(["false"]))  # e.g. a snap stub that cannot start here
            self.assertEqual(app._wait_for_window(["true"]), app.HANDED_OFF)  # passed to a browser already running
            self.assertEqual(app._wait_for_window(["sleep", "2.5"]), 0)  # a window the user closed

    def test_a_handed_off_window_does_not_report_a_close(self) -> None:
        with mock.patch.object(app, "ensure_server", return_value={"port": 1, "token": "t"}), \
                mock.patch.object(app, "notify_closed") as closed, mock.patch.object(app, "run_webkit", return_value=None), \
                mock.patch.object(app, "run_chromium", return_value=app.HANDED_OFF):
            self.assertEqual(app.main("auto"), 0)
        closed.assert_not_called()  # the close would pause training while the window is still open


class ProcessCheckTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ps"), "needs ps")
    def test_ps_recognises_live_workers_only(self) -> None:
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
        worker, other = subprocess.Popen([*sleeper, "model_builder", "worker"]), subprocess.Popen(sleeper)
        try:
            self.assertTrue(server._process_alive_ps(worker.pid))
            self.assertFalse(server._process_alive_ps(other.pid))
            worker.kill()  # not yet reaped: a zombie
            deadline = time.monotonic() + 5
            while server._process_alive_ps(worker.pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertFalse(server._process_alive_ps(worker.pid))
        finally:
            for process in (worker, other):
                process.kill()
                process.wait()
        self.assertFalse(server._process_alive_ps(worker.pid))


if __name__ == "__main__":
    unittest.main()
