import csv
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot3d", ROOT / "scripts/plot3d.py")
plot3d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot3d)


class PlotTests(unittest.TestCase):
    def render(self, directory, data, title="title", label="curve", *options):
        source, output = directory / "data.csv", directory / "plot.svg"
        source.write_text(data)
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/plot3d.py"), *options,
             str(output), title, "subtitle <&>", f"{source}:{label}:#e8b04b"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return ET.parse(output)

    def test_text_is_xml_escaped(self):
        with tempfile.TemporaryDirectory() as temp:
            tree = self.render(Path(temp), "0,1,0\n1,2,1\n", "A < B & C", "L <&>")
            text = "".join(tree.getroot().itertext())
            self.assertIn("A < B & C", text)
            self.assertIn("L <&>", text)
            self.assertIn("subtitle <&>", text)

    def test_constant_height_has_valid_ticks(self):
        with tempfile.TemporaryDirectory() as temp:
            tree = self.render(Path(temp), "0,1,0\n0,2,1\n")
            self.assertTrue(tree.getroot().tag.endswith("svg"))

    def test_errors_and_nonfinite_rows_break_the_curve(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "data.csv"
            source.write_text("0,1,0\n1,ERR,ERR\n2,2,0\n3,NaN,0\n4,1,0\n5,51,0\n")
            segments = plot3d.read_csv(source, None)
            self.assertEqual([len(segment) for segment in segments], [1, 1, 1])
            self.assertEqual([segment[0][0] for segment in segments], [0, 2, 4])

    def test_non_degenerate_singular_heights_are_not_bridged(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "data.csv"
            source.write_text("-3.01,1,0\n-2.99,2,0\n-2,3,0\n-1.99,4,0\n")
            segments = plot3d.read_csv(source, None, True)
            self.assertEqual([len(segment) for segment in segments], [1, 1, 1])
            self.assertEqual(len(plot3d.read_csv(source, None)[0]), 4)
            self.render(Path(temp), "-3.01,1,0\n-2.99,2,0\n-1.99,4,0\n",
                        "singularities", "curve", "--xrange=-4:0", "--break-negative-integers")


class ChartgenTests(unittest.TestCase):
    def fixture(self, directory, body):
        scripts = directory / "scripts"
        binary = directory / "target/release/tet"
        scripts.mkdir()
        binary.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "scripts/chartgen.sh", scripts / "chartgen.sh")
        binary.write_text("#!/usr/bin/env python3\nimport sys\n" + body)
        binary.chmod(0o700)
        return scripts / "chartgen.sh", directory / "result.csv"

    def run_sweep(self, script, output, multiplier):
        return subprocess.run(
            ["timeout", "15", "bash", str(script), "1", "0", str(output), multiplier],
            capture_output=True, text=True, timeout=20,
            env=dict(os.environ, SILENT="1", TET_MT="0"),
        )

    def test_invalid_multipliers_fail_without_running_tetration(self):
        for multiplier in ["0", "-1", "NaN", "Infinity", "not-a-number", "1e-1000"]:
            with self.subTest(multiplier=multiplier), tempfile.TemporaryDirectory() as temp:
                script, output = self.fixture(Path(temp), "raise AssertionError('must not run')\n")
                result = self.run_sweep(script, output, multiplier)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotEqual(result.returncode, 124, "invalid step hung")
                self.assertFalse(output.exists())

    def test_success_shape_and_singular_heights_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            script, output = self.fixture(Path(temp), "print('1\\n0')\n")
            result = self.run_sweep(script, output, "1000")
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = list(csv.reader(output.read_text().splitlines()))
            self.assertTrue(all(row[1:] == ["1", "0"] for row in rows))
            heights = [float(row[0]) for row in rows]
            self.assertEqual(heights, sorted(set(heights)))
            self.assertEqual(heights[0], -30)
            self.assertEqual(heights[-1], 120)
            self.assertIn(-2, heights)

    def test_failures_and_invalid_numeric_output_are_not_success(self):
        for body in [
            "print('intentional failure', file=sys.stderr)\nsys.exit(2)\n",
            "print('NaN\\n0')\n",
            "print('1')\n",
        ]:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as temp:
                script, output = self.fixture(Path(temp), body)
                result = self.run_sweep(script, output, "1000")
                self.assertNotEqual(result.returncode, 0)
                rows = list(csv.reader(output.read_text().splitlines()))
                self.assertTrue(all(row[1:] == ["ERR", "ERR"] for row in rows))
                self.assertTrue(Path(str(output) + ".errors.log").read_text())


if __name__ == "__main__":
    unittest.main()
