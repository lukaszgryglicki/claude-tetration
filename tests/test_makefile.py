import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MakefileTests(unittest.TestCase):
    def make(self, target=None, *variables, tool="make", dry=True, expected_error=None):
        environment = dict(os.environ, MAKEFLAGS="", MFLAGS="", MAKELEVEL="0")
        for name in [
            "CARGO", "RUSTC", "PYTHON", "JOBS", "CARGO_TARGET_DIR", "STATIC_TARGET",
            "BINDIR", "DESTDIR", "TEST_ARGS", "ARGS", "RUSTFLAGS",
        ]:
            environment.pop(name, None)
        command = [tool]
        if dry:
            command.append("-n")
        if target:
            command.append(target)
        result = subprocess.run(
            [*command, *variables], cwd=ROOT, env=environment,
            capture_output=True, text=True, timeout=15,
        )
        if expected_error is None:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(expected_error, result.stderr)
        return result.stdout

    def test_default_and_build_aliases_are_release(self):
        for target in [None, "all", "build", "release"]:
            with self.subTest(target=target):
                output = self.make(target)
                self.assertIn("cargo build", output)
                self.assertIn("--release", output)
                self.assertNotIn("cargo test", output)
                self.assertNotIn("chartgallery", output)
        self.assertNotIn("--release", self.make("debug"))

    def test_static_and_install_paths_are_separate(self):
        native_target = "$(rustc -vV | sed -n 's/^host: //p')"
        output = self.make("static", "CARGO_TARGET_DIR=build output")
        self.assertIn("-C target-feature=+crt-static", output)
        self.assertIn(f'target="{native_target}"', output)
        self.assertIn('--target "$target"', output)
        self.assertIn('--target-dir "build output"', output)
        self.assertIn(
            'target="custom-target"',
            self.make("static", "STATIC_TARGET=custom-target"),
        )
        for target, source in [
            ("install", "build output/release/tet"),
            ("install-static", f"build output/{native_target}/release/tet"),
        ]:
            with self.subTest(target=target):
                output = self.make(
                    target, "CARGO_TARGET_DIR=build output", "DESTDIR=/tmp/stage",
                )
                self.assertIn('install -d "/tmp/stage/data/scripts"', output)
                self.assertIn(f'install -m 0755 "{source}" "/tmp/stage/data/scripts/tet"', output)

    def test_static_driver_workaround_is_freebsd_only(self):
        with tempfile.TemporaryDirectory() as directory:
            cargo = Path(directory) / "cargo"
            cargo.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$RUSTFLAGS" "$@"\n'
                'printf "%s\\n" "stop after flags" >&2\nexit 1\n'
            )
            cargo.chmod(0o755)
            for target in ["x86_64-unknown-freebsd", "aarch64-unknown-freebsd",
                           "x86_64-unknown-linux-gnu"]:
                with self.subTest(target=target):
                    output = self.make(
                        "static", f"CARGO={cargo}", f"STATIC_TARGET={target}",
                        "RUSTFLAGS=-C opt-level=2", dry=False,
                        expected_error="stop after flags",
                    )
                    flags = output.splitlines()[0]
                    self.assertIn("-C opt-level=2 -C target-feature=+crt-static", flags)
                    self.assertEqual(
                        "-C link-arg=-Wno-unused-command-line-argument" in flags,
                        target.endswith("-freebsd"),
                    )
                    self.assertIn(f"--target\n{target}\n", output)

    def test_static_rejects_non_static_outputs(self):
        if not shutil.which("file"):
            self.skipTest("file is not installed")
        with tempfile.TemporaryDirectory() as directory:
            target_dir = Path(directory) / "build output"
            executable = target_dir / "fixture-target" / "release" / "tet"
            executable.parent.mkdir(parents=True)
            executable.write_text("This is not a native executable.\n")
            self.make(
                "static", "CARGO=true", "STATIC_TARGET=fixture-target",
                f"CARGO_TARGET_DIR={target_dir}", dry=False,
                expected_error="not a static executable",
            )

    def test_validation_targets_cover_existing_checks(self):
        for target, expected in [
            ("test", "cargo test"), ("test", "-p 'test_*.py'"),
            ("test-lib", "--lib"), ("test-doc", "--doc"),
            ("test-ignored", "--ignored"), ("test-all", "--include-ignored"),
            ("test-all", "-p 'test_*.py'"),
            ("test-st", "env -u TET_MT -u RAYON_NUM_THREADS"),
            ("fmt", "cargo fmt --all"), ("fmt-check", "cargo fmt --all -- --check"),
            ("vet", "--all-targets -- -D warnings"), ("lint", "cargo clippy"),
            ("check", "cargo check"), ("doc", "--no-deps"),
        ]:
            with self.subTest(target=target, expected=expected):
                self.assertIn(expected, self.make(target))
        output = self.make("verify")
        for command in ["cargo fmt", "cargo clippy", "cargo test", "-m unittest"]:
            self.assertIn(command, output)
        self.assertIn("-- --test-threads=1", self.make("test-rust"))
        self.assertIn(
            "-- string_inputs --test-threads=1",
            self.make("test-rust", "TEST_ARGS=string_inputs --test-threads=1"),
        )
        self.assertIn(
            "--doc -- --nocapture",
            self.make("test-doc", "TEST_ARGS=--nocapture"),
        )

    def test_example_and_script_arguments_are_forwarded(self):
        self.assertIn('"target/release/tet" 50 1.3 0.1 0.5 0', self.make("demo"))
        self.assertIn("--examples", self.make("examples"))
        for target, args, expected in [
            ("run", "--quiet 50 1.2 0 0.4 0.2", '"target/release/tet"'),
            ("grid", "--quiet 50 1 1.3 1.3 0.1 0.1 0.5 0.5 0 0",
             "--example grid_runner --"),
            ("edge", "--quiet 50", "--example edge_runner --"),
        ]:
            with self.subTest(target=target):
                self.assertIn(f"{expected} {args}", self.make(target, f"ARGS={args}"))
        self.assertIn("--example test_w0", self.make("test-w0"))
        output = self.make(
            "chartgen", "CARGO_TARGET_DIR=build output", 'ARGS=1 0 "plot output.csv" coarse',
        )
        self.assertIn('CARGO_TARGET_DIR="build output" bash scripts/chartgen.sh', output)
        self.assertIn('1 0 "plot output.csv" coarse', output)
        self.assertIn("scripts/plot3d.py --help", self.make("plot", "ARGS=--help"))
        self.assertIn("bash scripts/chartgallery.sh", self.make("gallery"))

    def test_clean_only_targets_cargo_outputs(self):
        output = self.make("clean", "CARGO_TARGET_DIR=isolated output")
        self.assertEqual(output.strip(), 'cargo clean --target-dir "isolated output"')

    def test_help_is_available_and_describes_targets(self):
        output = self.make("help", dry=False)
        for target in [
            "release", "debug", "static", "install", "demo", "test", "fmt",
            "vet", "clean", "grid", "edge", "chartgen", "plot", "gallery",
        ]:
            self.assertIn(target, output)
        self.assertIn("/data/scripts", output)

    def test_bsd_make_can_parse_the_same_targets(self):
        if not shutil.which("bmake"):
            self.skipTest("bmake is not installed")
        for target in ["all", "static", "test-all", "verify", "install", "help"]:
            with self.subTest(target=target):
                self.make(target, tool="bmake")


if __name__ == "__main__":
    unittest.main()
