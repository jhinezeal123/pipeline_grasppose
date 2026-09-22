#!/usr/bin/env python3
"""Install and exercise MinkowskiEngine inside the repo overlay, never the host."""
import os
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / '.venv'
REVISION = '02fc608bea4c0549b0a7b00ca1bf15dee4a0b228'
SOURCE = 'https://github.com/NVIDIA/MinkowskiEngine.git'
SMOKE = '''
import torch
import MinkowskiEngine as ME
coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
x = ME.SparseTensor(torch.ones((2, 2), device='cuda'), coordinates=coordinates, device='cuda')
layer = ME.MinkowskiConvolution(2, 3, kernel_size=1, dimension=3).cuda()
y = layer(x)
assert y.F.shape == (2, 3)
assert torch.isfinite(y.F).all()
torch.cuda.synchronize()
print('MinkowskiEngine CUDA convolution OK', ME.__version__)
'''


def pip_install(*args):
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(ENV / 'bin/python'),
                    'install', '-c', str(ENV / 'host-constraints.txt'), *args], check=True)


def probe():
    return subprocess.run([str(ENV / 'bin/python'), '-c', SMOKE],
                          capture_output=True, text=True)


def validate_cuda(nvcc_output, torch_cuda):
    match = re.search(r'release\s+(\d+\.\d+)', nvcc_output)
    if not match or match.group(1) != torch_cuda:
        raise RuntimeError(f'nvcc must match torch CUDA {torch_cuda}; select CUDA_HOME/PATH correctly.')


def prepare_setup(path):
    # Upstream setup.py unconditionally invokes bare pip uninstall. Remove only
    # that side effect in our disposable build copy; never run it against host pip.
    source = path.read_text()
    unwanted = 'run_command("pip", "uninstall", "MinkowskiEngine", "-y")'
    if source.count(unwanted) != 1:
        raise RuntimeError('Unexpected upstream setup.py; refusing to execute it.')
    path.write_text(source.replace(unwanted, '# Host uninstall disabled by pipeline installer.'))


def build_wheel():
    nvcc = (str(Path(os.environ['CUDA_HOME']) / 'bin/nvcc')
            if os.environ.get('CUDA_HOME') else shutil.which('nvcc'))
    if not nvcc or not Path(nvcc).is_file():
        raise RuntimeError('Source build needs nvcc. Set CUDA_HOME or provide MINKOWSKI_ENGINE_WHEEL.')
    cuda = subprocess.check_output([str(ENV / 'bin/python'), '-c',
                                   'import torch; print(torch.version.cuda)'], text=True).strip()
    validate_cuda(subprocess.check_output([nvcc, '--version'], text=True), cuda)
    if not shutil.which('git'):
        raise RuntimeError('Source build needs git.')
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    # Verify headers AND linkable BLAS before starting an expensive CUDA build.
    native = ENV / 'native'
    native.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='minkowski-', dir=native) as temp:
        temp = Path(temp)
        check = temp / 'blas.cpp'
        check.write_text('#include <cblas.h>\nint main(){return 0;}\n')
        try:
            subprocess.run([*compiler, str(check), '-lopenblas', '-o', str(temp / 'blas')],
                           check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError('Source build needs a C++ compiler and OpenBLAS headers/library '
                               '(Ubuntu: build-essential python3-dev libopenblas-dev), '
                               'or provide MINKOWSKI_ENGINE_WHEEL.') from exc
        # Python 3.8 on JetPack 5 cannot use current setuptools releases.
        pip_install('setuptools==68.2.2', 'wheel<0.46', 'ninja')
        source = temp / 'source'
        subprocess.run(['git', 'init', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'fetch', '--depth', '1', SOURCE, REVISION], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        prepare_setup(source / 'setup.py')
        build_env = os.environ.copy()
        build_env['CUDA_HOME'] = str(Path(nvcc).resolve().parents[1])
        build_env.setdefault('MAX_JOBS', '2')
        build_env.setdefault('TORCH_CUDA_ARCH_LIST', '7.2')
        log = ENV / 'minkowski-build.log'
        print(f'Building MinkowskiEngine {REVISION}; log: {log}', flush=True)
        with log.open('w') as output:
            result = subprocess.run([str(ENV / 'bin/python'), 'setup.py', 'bdist_wheel',
                                     '--force_cuda', '--blas=openblas'], cwd=source,
                                    env=build_env, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'MinkowskiEngine source build failed. See {log}; '
                               'a compatible MINKOWSKI_ENGINE_WHEEL may be needed for this Torch/CUDA.')
        wheels = list((source / 'dist').glob('*.whl'))
        if len(wheels) != 1:
            raise RuntimeError('Source build did not produce exactly one wheel.')
        pip_install('--no-deps', '--force-reinstall', str(wheels[0]))


# Evaluate tags inside the target interpreter: host Python can differ from .venv.
WHEEL_SELECTION = """
import json, sys
try:
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename, InvalidWheelFilename
except ImportError:
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename, InvalidWheelFilename
supported = set(sys_tags())
compatible = []
for path in json.loads(sys.argv[1]):
    from pathlib import Path
    try:
        _, _, _, tags = parse_wheel_filename(Path(path).name)
    except InvalidWheelFilename:
        continue
    if supported.intersection(tags):
        compatible.append(path)
print(json.dumps(compatible))
"""


def bundled_wheel():
    """Select one wheel matching target Python/ABI/platform, not Torch/CUDA ABI."""
    hits = sorted((ROOT / 'model').glob('minkowskiengine-*.whl'))
    if not hits:
        return None
    try:
        output = subprocess.check_output(
            [str(ENV / 'bin/python'), '-c', WHEEL_SELECTION,
             json.dumps([str(path) for path in hits])], text=True)
        compatible = [Path(path) for path in json.loads(output)]
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('Cannot determine target wheel tags; install packaging in .venv.') from exc
    if len(compatible) > 1:
        raise RuntimeError('Multiple compatible bundled wheels; set MINKOWSKI_ENGINE_WHEEL '
                           'explicitly because wheel tags do not identify Torch/CUDA ABI.')
    if not compatible:
        print('No bundled wheel matches target Python/ABI/platform; skipping bundled wheels.', flush=True)
        return None
    return compatible[0]


def main():
    if not (ENV / 'host-constraints.txt').is_file():
        raise RuntimeError('Run env/setup_env.py first.')
    wheel = os.environ.get('MINKOWSKI_ENGINE_WHEEL')
    initial = None
    if not wheel:
        initial = probe()
        if initial.returncode == 0:
            print(initial.stdout.strip())
            return  # Preserve a working extension, even when a wheel is bundled.
        local = bundled_wheel()
        if local:
            print(f'Dung wheel di kem repo: {local.name}', flush=True)
            wheel = str(local)
    if wheel:
        path = Path(wheel).expanduser().resolve()
        if path.suffix != '.whl' or not path.is_file():
            raise RuntimeError(f'MINKOWSKI_ENGINE_WHEEL must name an existing .whl file: {path}')
        pip_install('--no-deps', '--force-reinstall', str(path))
    else:
        result = initial
        # A broken installed extension is not a missing package. Keep its traceback
        # and require an explicit replacement instead of hiding the ABI error.
        present = subprocess.run([str(ENV / 'bin/python'), '-c',
                                  'import importlib.util, sys; '
                                  'sys.exit(0 if importlib.util.find_spec("MinkowskiEngine") else 1)'])
        if present.returncode != 1:
            raise RuntimeError('Existing MinkowskiEngine failed the CUDA check:\n' + result.stderr +
                               '\nProvide a compatible MINKOWSKI_ENGINE_WHEEL to replace it.')
        build_wheel()
    result = probe()
    if result.returncode:
        raise RuntimeError('Installed MinkowskiEngine failed the CUDA check:\n' + result.stderr)
    print(result.stdout.strip())


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f'MinkowskiEngine setup failed: {exc}')

