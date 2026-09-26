"""Metadata-only fixtures test release completeness; kernels are tested separately."""
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile
from check_release import EXPECTED, check


def make_candidate(root):
    with tarfile.open(root / 'fabba-1.5.2.tar.gz', 'w:gz') as archive:
        for name in ['pyproject.toml', 'setup.py', 'tests/test_extensions.py', 'tools/check_wheel.py'] + [f'fABBA/extmod/m{i}.pyx' for i in range(12)]:
            info = tarfile.TarInfo('fabba-1.5.2/' + name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b'\n'))
    platforms = {'linux-x86_64':'manylinux_2_28_x86_64', 'linux-aarch64':'manylinux_2_28_aarch64',
                 'macos-x86_64':'macosx_11_0_x86_64', 'macos-arm64':'macosx_11_0_arm64',
                 'windows-amd64':'win_amd64', 'windows-arm64':'win_arm64'}
    for py, platform in EXPECTED:
        with zipfile.ZipFile(root / f'fabba-1.5.2-{py}-{py}-{platforms[platform]}.whl', 'w') as archive:
            for i in range(12):
                archive.writestr(f'fABBA/extmod/m{i}.so', b'fixture')


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        make_candidate(self.root)

    def test_complete_candidate(self):
        check(self.root)

    def test_missing_platform_is_rejected(self):
        next(self.root.glob('*win_arm64.whl')).unlink()
        with self.assertRaisesRegex(ValueError, 'missing release wheels'): check(self.root)

    def test_missing_sdist_is_rejected(self):
        next(self.root.glob('*.tar.gz')).unlink()
        with self.assertRaisesRegex(ValueError, 'source distribution'): check(self.root)

    def test_mixed_versions_are_rejected(self):
        wheel = next(self.root.glob('*.whl'))
        wheel.rename(wheel.with_name(wheel.name.replace('1.5.2', '1.5.3')))
        with self.assertRaisesRegex(ValueError, 'mixed'): check(self.root)

    def test_nonportable_linux_wheel_is_rejected(self):
        wheel = next(self.root.glob('*manylinux_2_28_x86_64.whl'))
        wheel.rename(wheel.with_name(wheel.name.replace('manylinux_2_28', 'linux')))
        with self.assertRaisesRegex(ValueError, 'nonportable'): check(self.root)

    def test_unexpected_abi_is_rejected(self):
        wheel = next(self.root.glob('*cp312-cp312-*.whl'))
        wheel.rename(wheel.with_name(wheel.name.replace('cp312-cp312', 'cp312-abi3')))
        with self.assertRaisesRegex(ValueError, 'unexpected ABI'): check(self.root)

    def test_missing_extensions_are_rejected(self):
        wheel = next(self.root.glob('*.whl'))
        with zipfile.ZipFile(wheel, 'w'): pass
        with self.assertRaisesRegex(ValueError, '12 native'): check(self.root)


if __name__ == '__main__': unittest.main()
