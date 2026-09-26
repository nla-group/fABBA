Installation, native extensions and platform support
====================================================

For a release whose wheel checks have passed, pip selects a precompiled wheel
for the Python interpreter, OS and CPU architecture. **Installing that wheel
does not require the user to install Cython or a C compiler.** The wheel already
contains the compiled modules. Building from the source distribution does
require native build tools.

The local workflow defines the following release targets. This table describes
the required build/test matrix, not a claim that unpublished local changes have
already passed on every remote runner.

.. list-table:: Required wheel coverage (33 wheels per release)
   :header-rows: 1
   :widths: 24 20 20 36

   * - Platform
     - Architecture
     - CPython
     - Build and test environment
   * - macOS
     - Intel x86_64
     - 3.9–3.14
     - Native macos-15-intel runner
   * - macOS
     - Apple Silicon arm64
     - 3.9–3.14
     - Native macos-14 runner
   * - Linux with glibc
     - x86_64
     - 3.9–3.14
     - manylinux_2_28 on Ubuntu x86_64
   * - Linux with glibc
     - aarch64
     - 3.9–3.14
     - manylinux_2_28 on Ubuntu ARM64
   * - Windows
     - x64 (AMD64)
     - 3.9–3.14
     - Native Windows x64 runner and MSVC
   * - Windows
     - ARM64
     - 3.12–3.14
     - Native windows-11-arm runner and MSVC

These are standard CPython builds with the GIL, using 64-bit interpreters.
PyPy, free-threaded Python, 32-bit systems, Alpine/musl, and other CPU
architectures are outside this release wheel matrix. Do not interpret
``requires-python >= 3.9`` as a certification of every future interpreter ABI.

Linux wheels target glibc 2.28 or newer. macOS wheels request deployment target
11.0, but the full installation also depends on NumPy/SciPy and other packages:
their wheels can require a newer OS, especially on newer Python versions.
The native CI OS versions above are the tested baseline, not every historical
OS version capable of loading an individual fABBA binary. Windows must also
meet the selected CPython version's own OS requirements.

Install a tested release
------------------------

Use a virtual environment and a current pip::

   python -m venv .venv
   # macOS / Linux:
   source .venv/bin/activate
   # Windows PowerShell:
   # .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install --only-binary=:all: fABBA

The binary-only command intentionally fails if the package or one of its
dependencies has no compatible wheel. It does not silently start an unexpected
local compiler build. Run ``python -m pip debug --verbose`` to inspect the
platform tags accepted by the current interpreter. On Apple Silicon, an Intel
Python running under Rosetta selects Intel wheels, not native ARM wheels.

Check that native code is active
--------------------------------

This program performs a numerical roundtrip and displays the selected backend::

   import importlib
   import importlib.machinery
   import numpy as np
   from fABBA import fABBA

   implementation = importlib.import_module("fABBA.fabba")
   backend = importlib.import_module(implementation.compress.__module__)
   assert any(backend.__file__.endswith(suffix)
              for suffix in importlib.machinery.EXTENSION_SUFFIXES)
   x = np.sin(np.linspace(0, 6, 100))
   model = fABBA(tol=0, alpha=0, verbose=0)
   y = model.inverse_transform(model.fit_transform(x), start=x[0])
   np.testing.assert_allclose(y, x, atol=1e-12)
   print(backend.__file__)

The release workflow goes further: ``tools/check_wheel.py`` verifies active
backend dispatch, imports and exercises **all 12 Cython modules**, rejects
skipped compiled tests, and executes the four self-contained gallery scripts
outside the checkout.

Build from source
-----------------

For source development or an intentional local rebuild, first install:

* macOS: Xcode Command Line Tools (``xcode-select --install``).
* Windows: Visual Studio Build Tools with the **Desktop development with C++**
  workload and a Windows SDK, matching the Python architecture. MSVC is the
  compiler used by the release workflow.
* Debian/Ubuntu Linux: ``build-essential`` and the development headers for the
  selected Python (for system Python, typically ``python3-dev``).

Then install from a checkout or force the release source distribution::

   python -m pip install --upgrade pip
   python -m pip install .
   # Alternatively, rebuild a published version from its sdist:
   python -m pip install --no-binary=fABBA --no-cache-dir fABBA

PEP 517 build isolation installs setuptools, Cython and NumPy headers from
``pyproject.toml``. Do not use ``--no-build-isolation`` unless you deliberately
manage compatible build dependencies. Normal source builds fail visibly when
an extension cannot compile; they do not silently omit failing extensions.

An explicit compiler-free fallback remains available::

   # macOS / Linux:
   FABBA_NO_EXTENSIONS=1 python -m pip install .
   # Windows PowerShell:
   $env:FABBA_NO_EXTENSIONS = "1"
   python -m pip install .
   Remove-Item Env:FABBA_NO_EXTENSIONS

This fallback is not a compiled installation and is not published by the native
wheel workflow. Remove this environment setting before building native wheels.

NumPy ABI and Windows integer widths
------------------------------------

Build requirements use NumPy 2.x headers and Cython >= 3.1.3. The runtime NumPy
range is >= 1.24 and < 3. NumPy documents that extensions compiled against 1.x
cannot run on 2.x, while properly built 2.x extensions can also support 1.x.
The release workflow therefore retests the **actual candidate wheels** under
NumPy 1.24 on Python 3.9 and NumPy 1.26 on Python 3.12 for Linux, Windows x64
and macOS. Windows ARM64 uses NumPy 2.x wheels.

Index and label buffers use ``np.intp_t`` with explicit ``dtype=np.intp``.
They no longer assume that C ``long`` and NumPy's default integer have the same
width on Windows, Unix, NumPy 1.x and NumPy 2.x.

See :doc:`releasing` for the publication checklist, and the primary references:
`NumPy downstream compatibility guide <https://numpy.org/doc/stable/dev/depending_on_numpy.html>`_
and `cibuildwheel platform documentation <https://cibuildwheel.pypa.io/en/stable/platforms/>`_.
