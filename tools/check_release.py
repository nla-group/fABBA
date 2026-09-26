"""Reject incomplete or mixed-version release candidates; never uploads files."""
import argparse
from pathlib import Path
import tarfile
import zipfile
from packaging.utils import parse_sdist_filename, parse_wheel_filename

PYTHONS = {"cp39", "cp310", "cp311", "cp312", "cp313", "cp314"}
PLATFORMS = {"linux-x86_64", "linux-aarch64", "macos-x86_64", "macos-arm64", "windows-amd64"}
EXPECTED = {(py, platform) for py in PYTHONS for platform in PLATFORMS}
EXPECTED |= {(py, "windows-arm64") for py in ("cp312", "cp313", "cp314")}


def platform_key(tag):
    for arch in ("x86_64", "aarch64"):
        if tag.startswith("manylinux_") and tag.endswith("_" + arch):
            return "linux-" + arch
    for arch in ("x86_64", "arm64"):
        if tag.startswith("macosx_") and tag.endswith("_" + arch):
            return "macos-" + arch
    return {"win_amd64": "windows-amd64", "win_arm64": "windows-arm64"}.get(tag)


def check(directory):
    files = list(Path(directory).iterdir())
    sdists = [p for p in files if p.name.endswith(".tar.gz")]
    wheels = [p for p in files if p.suffix == ".whl"]
    if len(sdists) != 1:
        raise ValueError("expected exactly one source distribution")
    name, version = parse_sdist_filename(sdists[0].name)
    if name != "fabba":
        raise ValueError("source distribution must be fABBA")
    with tarfile.open(sdists[0]) as archive:
        names = archive.getnames()
        for required in ("pyproject.toml", "setup.py", "tests/test_extensions.py", "tools/check_wheel.py"):
            if not any(n.endswith("/" + required) for n in names):
                raise ValueError(f"sdist is missing {required}")
        if len([n for n in names if n.endswith(".pyx")]) != 12:
            raise ValueError("sdist must contain all 12 Cython sources")
    coverage = set()
    for wheel in wheels:
        wheel_name, wheel_version, _, tags = parse_wheel_filename(wheel.name)
        if wheel_name != name or wheel_version != version:
            raise ValueError(f"mixed project/version: {wheel.name}")
        if any(tag.abi != tag.interpreter for tag in tags):
            raise ValueError(f"unexpected ABI: {wheel.name}")
        identities = {(tag.interpreter, platform_key(tag.platform)) for tag in tags}
        if len(identities) != 1 or not identities <= EXPECTED:
            raise ValueError(f"unexpected or nonportable wheel: {wheel.name}")
        if coverage & identities:
            raise ValueError(f"duplicate platform/interpreter: {wheel.name}")
        with zipfile.ZipFile(wheel) as archive:
            binaries = [n for n in archive.namelist() if n.endswith((".so", ".pyd"))]
            if len(binaries) != 12:
                raise ValueError(f"wheel must contain 12 native modules: {wheel.name}")
        coverage |= identities
    if coverage != EXPECTED:
        raise ValueError(f"missing release wheels: {sorted(EXPECTED - coverage)}")
    print(f"Validated fABBA {version}: {len(wheels)} native wheels and one sdist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
