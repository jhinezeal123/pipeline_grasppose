"""Bootstrap regressions: no network, CUDA or third-party dependencies required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import venv

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('setup_env', ROOT / 'env/setup_env.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class EnvironmentTests(unittest.TestCase):
    def test_missing_native_stack_fails_before_install(self):
        with patch.object(setup.metadata, 'distributions', return_value=[]), \
                patch.object(setup.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Missing host torch'):
                setup.main()
            run.assert_not_called()

    def test_overlay_resolves_dependencies_without_touching_host(self):
        versions = {'torch': '2.10.0+cu128', 'torchvision': '0.25.0+cu128',
                    'numpy': '2.0.2', 'minkowskiengine': '0.5.4'}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(setup, 'ENV', Path(temp) / '.venv'), \
                patch.object(setup, 'protected_versions', return_value=versions), \
                patch.object(setup.subprocess, 'run') as run:
            setup.main()  # Create a real no-pip venv, but do not download anything.
            config = (setup.ENV / 'pyvenv.cfg').read_text()
            self.assertIn('include-system-site-packages = true', config)
            command = run.call_args.args[0]
            self.assertEqual(command[1:4], ['-m', 'pip', '--python'])
            self.assertEqual(command[4], str(setup.ENV / 'bin/python'))
            self.assertNotIn('--no-deps', command)
            self.assertIn('torch==2.10.0+cu128', (setup.ENV / 'host-constraints.txt').read_text())
            setup.main()  # Reuse the same venv safely.
            with patch.object(setup, 'protected_versions', return_value={**versions, 'torch': '2.11.0'}):
                with self.assertRaisesRegex(RuntimeError, 'changed'):
                    setup.main()

    def test_unmanaged_venv_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(setup, 'ENV', Path(temp)), \
                patch.object(setup, 'protected_versions', return_value={}), \
                patch.object(setup.subprocess, 'run'):
            with self.assertRaisesRegex(RuntimeError, 'unmanaged'):
                setup.main()

    def test_protects_cuda_packages_but_allows_ui_resolution(self):
        distributions = []
        for name in ('torch', 'torchvision', 'numpy', 'MinkowskiEngine',
                     'nvidia-cublas-cu12', 'triton', 'gradio'):
            distributions.append(Mock(metadata={'Name': name}, version='1.0'))
        with patch.object(setup.metadata, 'distributions', return_value=distributions):
            result = setup.protected_versions()
        self.assertIn('nvidia-cublas-cu12', result)
        self.assertIn('minkowskiengine', result)
        self.assertNotIn('gradio', result)


if __name__ == '__main__':
    unittest.main()
