Build and validate a PyPI release
=================================

The ``Release wheels`` GitHub Actions workflow builds artifacts; it does **not**
automatically publish to PyPI. Local changes remain unverified on platforms
whose native jobs have not run. Do not publish a candidate until all required
jobs pass.

Release sequence
----------------

1. Choose a new version in ``fABBA/__init__.py``. PyPI release files cannot be
   overwritten; do not reuse an already published filename. Keep release notes
   and compatibility declarations consistent with that version.
2. Commit the intended source, then run ``Release wheels`` manually or push its
   ``v*`` version tag when ready. Pull requests touching relevant sources also
   run the wheel workflow. No remote dispatch occurs merely by editing locally.
3. The workflow builds and checks one source distribution. Every native job
   consumes that exact archive, so missing Cython sources or test files are
   detected before publication.
4. cibuildwheel builds 33 standard CPython wheels across the six native runner
   targets. Linux wheels use manylinux containers and auditwheel repair;
   macOS uses delocate. Binary dependencies must be available as wheels during
   testing. There are no test skips for declared native targets.
5. Each wheel must import and execute all extensions, select native backends,
   pass the numerical suite and run the example programs. Separate NumPy 1.x
   jobs install the actual candidate wheels under pinned older dependencies.
6. The final job rejects missing platforms, mixed versions, duplicate targets,
   nonportable Linux tags, and wheels lacking their 12 extensions. It also runs
   ``twine check --strict``. Only then is ``pypi-release-candidate`` uploaded
   as a GitHub Actions artifact.
7. Download that **complete candidate**, extract it into a clean local
   directory and publish the sdist plus all wheels together using the normal
   authenticated PyPI process. Never publish just the one wheel built on your
   development machine.

Review before uploading
-----------------------

From the matching source checkout, with the downloaded candidate in
``release-candidate/``::

   python -m pip install --upgrade twine packaging
   python tools/check_release.py release-candidate
   python -m twine check --strict release-candidate/*
   # Run only when you intentionally want to publish this reviewed release:
   python -m twine upload release-candidate/*

Store authentication in your normal credential manager or use a separately
configured PyPI Trusted Publisher. Credentials are not stored in this repository.
The local maintenance task neither publishes nor configures a publisher.

After publication
-----------------

Test installation of the exact version in fresh environments on the supported
platforms. Use binary-only installation to ensure PyPI serves the expected
wheels, then execute the backend check in :doc:`installation`. Keep the build
logs and candidate artifacts associated with the release. A passing metadata
check alone is not proof that the numerical extensions load.

Limits and extending support
----------------------------

See :doc:`installation` for the OS/architecture/Python matrix and dependency
limits. Adding a target requires a native build/test job, available binary
dependencies, and an update to both the cibuildwheel selectors and
``tools/check_release.py``. Do not suppress a failing job or allow an empty
matrix just to produce a green release. New Python ABIs, free-threaded Python,
Alpine/musl and 32-bit targets require separate validation before being claimed.
