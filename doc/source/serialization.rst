Inspect, export and restore a codebook
======================================

Retain these items together:

* Symbol sequence, including the exact Unicode characters.
* Starting value in the same units used during fitting.
* ``parameters.centers`` and ``parameters.alphabets``.
* Original sample count, fitting configuration and package version for provenance.
* Any external normalization statistics, timestamps or channel metadata.

``Model.to_dict()`` returns an independent JSON-compatible object with
``schema_version=1``. ``Model.from_dict()`` validates finite centers, positive
lengths and a unique single-character alphabet. It rejects unsupported schema
versions. This schema covers the univariate ``fABBA`` codebook; JABBA and QABBA
have different model objects and are not interchangeable with it.

Complete portable roundtrip
---------------------------

.. literalinclude:: ../../example/export_codebook.py
   :language: python

The example uses a temporary directory. To keep the export, use a persistent
``Path("signal.json")``. Decoding with an explicit restored codebook does not
require ``fit`` and does not restore the original training signal. The JSON
stores a lossy representation, not residuals.

A codebook is not a trained nearest-center encoder
--------------------------------------------------

The univariate ``fABBA`` class has ``fit`` and ``fit_transform``. Refitting learns
a new codebook. For encoding held-out series against fixed training centers,
use ``JABBA.transform`` as shown in :doc:`multivariate`.

Legacy pickle support
---------------------

``model.dump(path)`` saves only the codebook. ``model.load(path)`` returns it;
``model.load(path, replace=True)`` installs it in that estimator. These methods
do not retain symbols, starting values or preprocessing metadata. Pickle can
execute code during loading: only load trusted files. JSON is preferable for
inspection and exchange; neither format should be assumed compatible with
future schema changes without checking its version.
