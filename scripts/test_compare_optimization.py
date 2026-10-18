"""Check benchmark accounting without CUDA or model loading."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('compare_optimization.py')


class ComparisonAccounting(unittest.TestCase):
    def execute(self, output, changed_ids=False, extra=None, gpu_csv=None):
        calls = []

        def fake_run(command, **_):
            index = len(calls)
            calls.append(command)
            count = int(command[-1])
            speed = 104 if index in (1, 2) else 100
            rows = []
            for row in range(count):
                rate = 200 if row < 2 else speed
                rows.append({'decode_tokens': 23, 'decode_ms': 23000 / rate,
                             'ttft_ms': 2, 'prompt_ids': [2, 7],
                             'generated_ids': [9 if changed_ids and index == 1 else 8]})
            return types.SimpleNamespace(stdout='\n'.join('BENCH_SAMPLE ' + json.dumps(r) for r in rows))

        argv = [str(SCRIPT), '--baseline', 'baseline.exe', '--output', str(output),
                '--runs', '3', '--discard-leading', '2', '--abba'] + (extra or [])
        with patch.object(sys, 'argv', argv), patch('subprocess.run', fake_run), \
                patch('subprocess.check_output', return_value=gpu_csv), \
                patch.dict(os.environ, {'ADAFORGE_EXPERIMENTAL_Q8': '0'}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(str(SCRIPT), run_name='__main__')
        return calls

    def test_abba_discards_warm_requests_and_preserves_raw_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            calls = self.execute(output)
            report = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(c[-1] == '5' for c in calls))
        labels = ['before', 'after', 'after_repeat', 'before_repeat']
        self.assertEqual(report['protocol']['sequence'], labels)
        self.assertEqual(report['protocol']['warmups'], 3)
        for label in labels:
            self.assertEqual(len(report[label]['discarded_samples']), 2)
            self.assertEqual(len(report[label]['samples']), 3)
        self.assertAlmostEqual(report['comparison']['relative_delta_percent'], 4)
        self.assertEqual(report['comparison']['control_drift_percent'], 0)

    def test_boundary_telemetry_does_not_change_timed_sample_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            calls = self.execute(output, extra=['--gpu-telemetry'], gpu_csv='RTX 4090, 44, 73 W, 2625 MHz, 10251 MHz, 0 %, 3000 MiB\n')
            report = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(len(calls), 4)
        self.assertTrue(report['protocol']['gpu_telemetry'])
        for label in report['protocol']['sequence']:
            self.assertEqual(len(report[label]['samples']), 3)
            for boundary in ['before', 'after']:
                self.assertIn('2625 MHz', report[label]['gpu_telemetry'][boundary]['raw_csv'])
        self.assertAlmostEqual(report['comparison']['relative_delta_percent'], 4)

    def test_changed_tokens_reject_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            with self.assertRaises(AssertionError):
                self.execute(output, changed_ids=True)
            self.assertFalse(output.exists())

    def test_total_requests_must_fit_cli_limit(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(SystemExit) as failure:
            self.execute(Path(directory) / 'result.json', extra=['--runs', '20'])
        self.assertEqual(failure.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
