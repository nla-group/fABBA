# Local maintenance record — 2026-09-26

All changes were made locally. No commit, push, pull request, release or remote
configuration change was made.

## Changes

- Replaced the obsolete `fabba_model`/pickle-fixture test runner with discoverable
  `unittest` contracts. Historical fixtures and experiments remain available.
- Added a GitHub Actions matrix for Linux, macOS and Windows; Python 3.9, 3.12,
  3.13; and compiled/Python backends. Jobs build an sdist and wheel, install the
  wheel outside the checkout, verify the backend and run tests and examples.
- Fixed Docker's invalid multiline JSON entry point and installed the actual
  package plus its compiler requirements. Added a Docker build context ignore file.
- Consolidated package metadata in `pyproject.toml`, with a literal version read
  from the package, current license metadata and declared test/docs extras.
  `FABBA_NO_EXTENSIONS=1` selects a compiler-free source build. Build paths are
  separated by backend, and prebuilt shared objects/bytecode are excluded from
  source/package data.
- Added shared shape, real-value, finiteness and parameter validation; fixed NaN
  boundary filling, input mutation, constant-feature division by zero, custom
  alphabets at exact capacity and misleading not-fitted exceptions.
- Unified the estimator and functional digitizers and their codebook class.
  Shared public univariate reconstruction now rounds cumulative lengths without
  mutating input and prevents zero-duration segments at half-grid boundaries.
- Fixed fABBA partition boundaries/remainders and JABBA remainder splitting and
  repeated single-series inference splitting. Added shared-codebook validation.
- Added `Model.to_dict()` / `Model.from_dict()` (JSON schema version 1), useful
  fitted attributes and working parameter inspection. Pickle handles are closed
  deterministically; documentation explains its trusted-input requirement.
- Removed large dead implementation blocks and global warning suppression.
  Deferred image-library import until image loading, avoiding plotting startup
  and font scans during ordinary package import.
- Reworked the Sphinx user guide and API reference. Added architecture, testing,
  serialization and application guides; four executable scripts cover five toy
  signals, parameter sweeps, portable JSON reconstruction and training-only
  JABBA codebooks for a toy classifier. Adjusted README badges and entry points.

## Compatibility notes

- This checkout now declares Python >= 3.9, NumPy >= 1.24, SciPy >= 1.9 and
  scikit-learn >= 1.2. Python 3.8 is outside the updated support declaration.
- Empty/single-sample signals, infinite values, ambiguous matrices and invalid
  parameters are rejected. Row/column vectors remain accepted by fABBA.
- Zero `tol` and `alpha` are supported. Exact-limit tests use numerical tolerances,
  not bitwise equality. Full symbolic RMSE is not universally monotone.
- Filling/quantization return copies. Unknown symbols raise `ValueError`.
  Validated public reconstruction can differ from historical rounding results.
- Partition fixes change results that previously dropped data. JABBA's independent
  chunks and fABBA's overlapping polygonal chunks have different semantics.
- The univariate JSON schema does not serialize JABBA/QABBA model objects.
- Original tracked generated files were retained; new artifacts are ignored.
  Existing CRLF line endings were preserved in modified legacy files.

## CI diagnosis and verification limits

The latest inspected failure was GitHub Pages run
[28744936821](https://github.com/nla-group/fABBA/actions/runs/28744936821).
Its build and artifact upload succeeded. Deployment failed during file syncing
with `Deployment failed, try again later`. No Python traceback or Sphinx failure
was reported in that job. A local code change cannot verify hosting recovery;
that deployment needs a remote retry when publication is desired.

The new workflow matrix has not been run remotely because nothing was pushed.
Docker is not installed on this host: the Dockerfile was statically checked,
but an image build was not executed locally. Windows/Linux, Python 3.9, GPU
backends and all research variants are not locally certified by these checks.
Compiler warnings in legacy generated kernels remain; successful compilation
is not a claim that every optional variant has been exhaustively validated.

## Reproduce the principal checks

```bash
python -m pip install -e ".[test,docs]"
python -m unittest discover -s tests -v
python -m build
python -m sphinx -W --keep-going -b html doc/source doc/_build/html
python -m sphinx -W --keep-going -b doctest doc/source doc/_build/doctest
python example/toy_models.py
python example/export_codebook.py
python example/tolerance_sweep.py
python example/shared_codebook.py
```

`runtest.py` is an alternative entry point to the same unittest suite.
For a Python-only build, set `FABBA_NO_EXTENSIONS=1` before installation and
use a fresh environment. Test installed artifacts outside the source tree.

## Verified results

| Check | Result |
| --- | --- |
| macOS arm64, Python 3.12.5 / NumPy 1.26.4, installed Python-only wheel | 22 unittest tests: 21 passed, 1 expected compiled-backend skip |
| macOS arm64, Python 3.13.1 / NumPy 2.5.3 / scikit-learn 1.9.1, compiled wheel built from the final sdist | All 22 unittest tests passed, including compiled/Python compression parity |
| Legacy `python runtest.py` entry point, Python 3.12 | Same 22-test suite passed with the expected Python-only skip |
| Four gallery scripts, both installed backends outside the checkout | All passed; toy frequency predictions were [0, 1] |
| Sphinx HTML, warnings as errors | Passed |
| Sphinx doctest, warnings as errors | 41 examples passed, zero failures |
| Packaging | Source distribution, Python-only wheel and macOS compiled wheels built successfully |
| Python-only wheel contents | No .so, .pyd or .pyc files; validation module included |
| Source distribution contents | Tests, executable examples and new documentation guides included |
| Workflow YAML / Docker entry point | Parsed successfully; Docker not executed locally |
| Whitespace review | Passed with `git -c core.whitespace=cr-at-eol diff --check`, respecting existing CRLF files |

The zero-tolerance sweep reconstructed the deterministic test signal to numerical
precision. At (tol, alpha) = (0.1, 0.5), its measured polygonal/symbolic RMSEs
were approximately 0.2884/0.346224; at (0.001, 0.01), both were approximately
0.0275682. These are measurements on one toy signal, not universal error bounds.

## Follow-up: cross-platform PyPI artifacts

- Added a separate release workflow that builds one sdist and consumes that
  exact archive on six native runner targets. It requires 33 CPython wheels:
  3.9–3.14 on Linux x86_64/aarch64, macOS Intel/ARM64 and Windows x64, plus
  3.12–3.14 on Windows ARM64. The ordinary test matrix also includes Python 3.14.
- Build dependencies now use NumPy 2.x headers and Cython >= 3.1.3. The same
  candidate wheels are retested with pinned NumPy 1.x environments, rather than
  rebuilding a different binary for the compatibility check.
- Replaced platform-dependent C integer buffers with `np.intp_t` and explicit
  `dtype=np.intp` in seven aggregation sources. Fixed unchecked negative
  indexing in both one-dimensional JABBA aggregation kernels.
- Added four native-kernel tests covering all 12 compiled modules. The complete
  numerical suite now contains 26 tests. Installed-wheel checks reject native
  backend fallback and skipped tests, then run all four gallery scripts.
- Added candidate checks for all required platforms/ABIs, matching versions,
  duplicate targets, portable platform tags, complete Cython sources and native
  module counts. Seven metadata-fixture unit tests exercise this gate. These
  fixtures are not evidence that 33 real wheels have been built locally.
- Added installation and release guides covering compiler prerequisites,
  compiler-free wheel installation, source builds, explicit Python fallback,
  NumPy ABI compatibility and the precise support limits.
- No PyPI upload job was added. A complete `pypi-release-candidate` artifact is
  assembled only after all required native build, runtime and metadata jobs pass.

### Additional local evidence

| Check | Result |
| --- | --- |
| macOS ARM64 / Python 3.12.5: compiled wheel built from sdist using NumPy 2.0.2 | All 12 native extensions built |
| That identical wheel under NumPy 1.26.4 | All 26 tests and four scripts passed |
| That identical wheel under NumPy 2.0.2 | All 26 tests and four scripts passed |
| macOS ARM64 / Python 3.14.5 / NumPy 2.5.3: fresh compiled installation from sdist | All 12 native extensions built; all 26 tests and four scripts passed |
| Release metadata unit tests | All seven passed; an incomplete local candidate was also rejected |
| cibuildwheel target enumeration | 33 unique build identifiers across six configured native targets |
| Strict Twine metadata checks | Local sdist and compiled wheel passed |
| Updated Sphinx guides | Strict HTML build passed; all 41 doctests passed |
| Updated workflow YAML | Parsed successfully |

Windows/Linux native execution and the complete remote 33-wheel matrix remain
release prerequisites, not locally verified results. The matrix excludes musl,
32-bit, PyPy and free-threaded interpreters. macOS deployment targets do not
override the minimum OS requirements of dependencies. No commit, push, PR,
workflow dispatch or publication was performed.
