import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('install_minkowski', ROOT / 'env/install_minkowski.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class MinkowskiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = Path(self.temp.name)
        # ROOT phai la thu muc RIENG trong temp, khong dung chung thu muc cha:
        # tempfile dat moi thu muc tam canh nhau, nen self.env.parent chinh la
        # /tmp — va mot test de lai /tmp/model/*.whl se lam test sau nhin thay
        # no. Da gap dung the: wheel con sot lai tu lan chay truoc.
        self.root = self.env / 'repo'
        (self.root / '.venv').mkdir(parents=True)
        (self.root / '.venv' / 'host-constraints.txt').write_text('torch==2.10.0+cu128\n')
        patcher = patch.object(installer, 'ENV', self.root / '.venv')
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(installer, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.dict(installer.os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_reuses_working_extension_without_installing(self):
        with patch.object(installer, 'probe', return_value=Mock(returncode=0, stdout='OK')), \
                patch.object(installer, 'pip_install') as pip, \
                patch.object(installer, 'build_wheel') as build:
            installer.main()
            pip.assert_not_called()
            build.assert_not_called()

    def test_missing_extension_builds_then_checks_cuda(self):
        with patch.object(installer, 'probe', side_effect=[Mock(returncode=1), Mock(returncode=0, stdout='OK')]) as probe, \
                patch.object(installer.subprocess, 'run', return_value=Mock(returncode=1)), \
                patch.object(installer, 'build_wheel') as build:
            installer.main()
            build.assert_called_once()
            self.assertEqual(probe.call_count, 2)

    def test_broken_existing_extension_keeps_error(self):
        with patch.object(installer, 'probe', return_value=Mock(returncode=1, stderr='undefined symbol')), \
                patch.object(installer.subprocess, 'run', return_value=Mock(returncode=0)), \
                patch.object(installer, 'build_wheel') as build:
            with self.assertRaisesRegex(RuntimeError, 'undefined symbol'):
                installer.main()
            build.assert_not_called()

    def test_explicit_wheel_installed_without_changing_torch(self):
        wheel = self.env / 'MinkowskiEngine-0.5.4-cp312-cp312-linux_x86_64.whl'
        wheel.touch()
        with patch.dict(installer.os.environ, {'MINKOWSKI_ENGINE_WHEEL': str(wheel)}), \
                patch.object(installer, 'pip_install') as pip, \
                patch.object(installer, 'probe', return_value=Mock(returncode=0, stdout='OK')):
            installer.main()
            pip.assert_called_once_with('--no-deps', '--force-reinstall', str(wheel))

    def test_bad_wheel_fails_cuda_check(self):
        wheel = self.env / 'bad.whl'
        wheel.touch()
        with patch.dict(installer.os.environ, {'MINKOWSKI_ENGINE_WHEEL': str(wheel)}), \
                patch.object(installer, 'pip_install'), \
                patch.object(installer, 'probe', return_value=Mock(returncode=1, stderr='CPU only')):
            with self.assertRaisesRegex(RuntimeError, 'CPU only'):
                installer.main()

    def test_missing_wheel_does_not_fallback_silently(self):
        with patch.dict(installer.os.environ, {'MINKOWSKI_ENGINE_WHEEL': '/missing.whl'}), \
                patch.object(installer, 'build_wheel') as build:
            with self.assertRaisesRegex(RuntimeError, 'existing .whl'):
                installer.main()
            build.assert_not_called()

    def test_cuda_mismatch_rejected(self):
        installer.validate_cuda('Cuda compilation tools, release 12.8, V12.8.93', '12.8')
        with self.assertRaisesRegex(RuntimeError, 'must match'):
            installer.validate_cuda('release 12.1, V12.1', '12.8')

    def test_upstream_uninstall_removed_before_build(self):
        setup = self.env / 'setup.py'
        setup.write_text('run_command("pip", "uninstall", "MinkowskiEngine", "-y")\nsetup()\n')
        installer.prepare_setup(setup)
        self.assertNotIn('run_command(', setup.read_text())
        self.assertIn('setup()', setup.read_text())
        with self.assertRaisesRegex(RuntimeError, 'Unexpected'):
            installer.prepare_setup(setup)

    def test_pip_targets_overlay_and_constraints(self):
        with patch.object(installer.subprocess, 'run') as run:
            installer.pip_install('--no-deps', 'package.whl')
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--python') + 1], str(self.root / '.venv' / 'bin/python'))
            self.assertEqual(cmd[cmd.index('-c') + 1], str(self.root / '.venv' / 'host-constraints.txt'))

    def test_bundled_wheel_preferred_over_source_build(self):
        """Wheel di kem repo phai duoc dung thay vi build source.

        MinkowskiEngine khong build duoc voi Torch 2.10/CUDA 12.8 (da do tren
        Kaggle: cung box, pointnet2._ext build duoc con no thi khong), nen khi
        repo co wheel san thi KHONG duoc thu build source nua.
        """
        model = self.root / 'model'
        model.mkdir(exist_ok=True)
        wheel = model / 'minkowskiengine-0.5.4-cp312-cp312-linux_x86_64.whl'
        wheel.touch()
        with patch.object(installer, 'pip_install') as pip, \
                patch.object(installer, 'probe', side_effect=[Mock(returncode=1, stderr='missing'), Mock(returncode=0, stdout='OK')]), \
                patch.object(installer, 'bundled_wheel', return_value=wheel), \
                patch.object(installer, 'build_wheel') as build:
            installer.main()
            pip.assert_called_once_with('--no-deps', '--force-reinstall', str(wheel))
            build.assert_not_called()

    def test_env_var_wheel_overrides_bundled(self):
        """MINKOWSKI_ENGINE_WHEEL phai thang wheel di kem repo."""
        model = self.root / 'model'
        model.mkdir(exist_ok=True)
        (model / 'minkowskiengine-0.5.4-cp312-cp312-linux_x86_64.whl').touch()
        chosen = self.env / 'chosen.whl'
        chosen.touch()
        with patch.dict(installer.os.environ, {'MINKOWSKI_ENGINE_WHEEL': str(chosen)}), \
                patch.object(installer, 'pip_install') as pip, \
                patch.object(installer, 'probe', return_value=Mock(returncode=0, stdout='OK')):
            installer.main()
            pip.assert_called_once_with('--no-deps', '--force-reinstall', str(chosen))

    def test_healthy_install_ignores_even_incompatible_bundled_wheel(self):
        model = self.root / 'model'
        model.mkdir()
        (model / 'minkowskiengine-0.5.4-cp39-cp39-linux_x86_64.whl').touch()
        with patch.object(installer, 'probe', return_value=Mock(returncode=0, stdout='OK')), \
                patch.object(installer, 'pip_install') as pip, \
                patch.object(installer, 'bundled_wheel') as select:
            installer.main()
            pip.assert_not_called()
            select.assert_not_called()

    def test_incompatible_bundle_falls_back_when_missing(self):
        with patch.object(installer, 'probe', side_effect=[Mock(returncode=1), Mock(returncode=0, stdout='OK')]), \
                patch.object(installer, 'bundled_wheel', return_value=None), \
                patch.object(installer.subprocess, 'run', return_value=Mock(returncode=1)), \
                patch.object(installer, 'pip_install') as pip, \
                patch.object(installer, 'build_wheel') as build:
            installer.main()
            pip.assert_not_called()
            build.assert_called_once()

    def test_real_target_tags_skip_wrong_python_and_platform(self):
        # Execute real packaging tag logic, without pip installation or a GPU.
        # A target interpreter wrapper ensures we use ENV/bin/python, not host tags.
        import venv
        venv.EnvBuilder(system_site_packages=True, with_pip=False).create(self.root / '.venv')
        model = self.root / 'model'
        model.mkdir()
        wrong_python = 'cp310' if sys.version_info[:2] == (3, 12) else 'cp312'
        for name in (f'minkowskiengine-0.5.4-{wrong_python}-{wrong_python}-linux_x86_64.whl',
                     'minkowskiengine-0.5.4-cp312-cp312-win_arm64.whl'):
            (model / name).touch()
        self.assertIsNone(installer.bundled_wheel())
        # A universal wheel is used only as a tag-selection fixture; no install.
        selected = model / 'minkowskiengine-0.5.4-py3-none-any.whl'
        selected.touch()
        self.assertEqual(installer.bundled_wheel(), selected)
        (model / 'minkowskiengine-0.5.5-py3-none-any.whl').touch()
        with self.assertRaisesRegex(RuntimeError, 'Multiple compatible'):
            installer.bundled_wheel()

    def test_host_interpreter_remains_absolute_after_path_change(self):
        # Exercise the shell assignment used by run.sh, including an executable
        # whose directory has spaces. Later .venv PATH changes must not redirect it.
        host = self.env / 'host python'
        overlay = self.env / 'overlay'
        host.mkdir()
        overlay.mkdir()
        for directory, label in ((host, 'host'), (overlay, 'overlay')):
            executable = directory / 'python3'
            executable.write_text('#!/bin/sh\necho ' + label + '\n')
            executable.chmod(0o755)
        line = next(line for line in (ROOT / 'run.sh').read_text().splitlines()
                    if line.startswith('HOST_PYTHON='))
        result = subprocess.check_output(['/bin/bash', '-c',
            'PATH="$1"\n' + line + '\nPATH="$2:$PATH"\n"$HOST_PYTHON"',
            'test', str(host), str(overlay)], text=True)
        self.assertEqual(result.strip(), 'host')

