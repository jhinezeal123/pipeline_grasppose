import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]; spec=importlib.util.spec_from_file_location('setup_env',ROOT/'env/setup_env.py'); setup=importlib.util.module_from_spec(spec); spec.loader.exec_module(setup)
class EnvironmentTests(unittest.TestCase):
    def test_requires_jetpack_stack(self):
        with patch.object(setup.metadata,'distributions',return_value=[]):
            with self.assertRaisesRegex(RuntimeError,'Missing host torch'): setup.protected_versions()
    def test_does_not_pin_ui_packages(self):
        d=[Mock(metadata={'Name':n},version='1.0') for n in ('torch','torchvision','numpy','gradio','ultralytics')]
        with patch.object(setup.metadata,'distributions',return_value=d): v=setup.protected_versions()
        self.assertIn('torch',v); self.assertNotIn('gradio',v); self.assertNotIn('ultralytics',v)
    def test_install_uses_overlay_python_and_constraints(self):
        versions={'torch':'2.1.0','torchvision':'0.16.0','numpy':'1.24.4'}
        with tempfile.TemporaryDirectory() as temp, patch.object(setup,'ENV',Path(temp)/'.venv'), patch.object(setup,'protected_versions',return_value=versions), patch.object(setup,'_create_overlay') as create, patch.object(setup.subprocess,'run') as run, patch.object(setup.sys,'argv',['setup_env.py','--install']):
            def make_env(): setup.ENV.mkdir(parents=True); (setup.ENV/'bin').mkdir()
            create.side_effect=make_env; setup.main(); cmd=run.call_args_list[-1].args[0]
            self.assertEqual(cmd[:3],[str(setup.ENV/'bin/python'),'-m','pip']); self.assertIn(str(setup.ENV/'host-constraints.txt'),cmd); self.assertIn(str(setup.ROOT/'requirements.txt'),cmd)
if __name__=='__main__': unittest.main()
