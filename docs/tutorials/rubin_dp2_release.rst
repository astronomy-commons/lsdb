Rubin Early Data Preview 2 (EDP2) HATS Catalogs
================================================================

This page describes the HATS distribution of Rubin's **Early Data Preview 2
(EDP2)** — the early release of the Rubin Data Preview 2 (DP2) data products,
repackaged into the `HATS <https://hats.readthedocs.io>`_ format for use with
LSDB. In addition to mirroring the Rubin Data Release  catalogs, the HATS distribution adds a number of value-added columns
intended to make common science workflow easier.

What's in the release
------------------------------------------------------

The EDP2 HATS distribution provides three catalogs, all of which can be opened with LSDB:

* ``object_collection`` — the DRP ``Object`` table with its nested
  ``objectForcedSource`` light curves.
* ``dia_object_collection`` — the DRP ``DiaObject`` table with its nested
  ``diaSource`` and ``diaObjectForcedSource`` light curves.
* ``object_photoz`` — photometric-redshift estimates for EDP2 objects
  (see :ref:`object_photoz <edp2-photoz>` below).

The build also publishes an auxiliary visit–detector metadata table under
``public-files/`` (see :ref:`Visit–detector metadata table <edp2-visit-detector>`
below).

These catalogs are available to Rubin data-rights holders. For step-by-step
instructions on accesing them see the
:doc:`Accessing Rubin Data Preview 2 tutorial </tutorials/pre_executed/rubin_dp2>`.


.. _edp2-nested-lightcurves:

Light curves are nested within their objects
------------------------------------------------------

Each catalog is a single table with **one row per object**, and every object
carries its own light curve as a *nested* column — the per-epoch measurements are
stored inline with the parent object rather than in a separate source table that
you would have to join yourself. LSDB loads these as a
:doc:`NestedFrame </tutorials/pre_executed/nestedframe>` (backed by
`nested-pandas <https://nested-pandas.readthedocs.io>`_), so the light curves
travel with their objects through filtering, cross-matching, and analysis.

* ``object_collection`` — each ``Object`` row (indexed by ``objectId``) has a
  nested ``objectForcedSource`` column holding its forced-photometry light curve.
* ``dia_object_collection`` — each ``DiaObject`` row (indexed by ``diaObjectId``)
  has nested ``diaSource`` and ``diaObjectForcedSource`` columns.

Under the hood a nested column is a *struct of lists*: each sub-column (for
example ``psfFlux``, ``psfMag``, ``band``, ``midpointMjdTai``) is a list with one entry per epoch, all aligned to the same parent row. 
You can keep the light curves nested and operate on them in place, select an
individual object's light curve, or flatten a nested column back into a flat
per-epoch source table — see
:doc:`Exploding Lightcurves to flat Source Tables </tutorials/pre_executed/explode_lightcurves>`.


How the HATS catalogs differ from the standard catalogs
------------------------------------------------------------------

Relative to the standard Rubin data products, the HATS ``object_collection``
and ``dia_object_collection`` catalogs add derived columns and downcast a subset
of columns. The changes are summarized below.

.. warning::

    **Do not use the downcast position columns where full precision matters.**
    The per-band ``{band}_ra`` and ``{band}_dec`` columns are stored as
    ``float32``. The float32 rounding error can be **larger** than the
    corresponding ``{band}_raErr`` / ``{band}_decErr`` values, so these columns
    are unsuitable for precision astrometry or tight positional cross-matching.
    Use the full-precision object position for those tasks.


Changes common to both collections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Magnitude and magnitude-error columns.** Magnitude and magnitude-error columns
converted from the flux columns are added. The magnitude error is defined as
half the difference between the magnitudes corresponding to ``Flux + FluxErr``
and ``Flux - FluxErr``. The full list of magnitude columns added for each
collection is given in the per-collection sections below.

**Corrected flux-error subcolumns.** Four subcolumns are added to the nested
forced-source columns (``objectForcedSource`` and ``diaObjectForcedSource``):

* ``psfFluxErr_corrected``
* ``psfDiffFluxErr_corrected``
* ``psfFluxErr_corrected_flag``
* ``psfDiffFluxErr_corrected_flag``

These are produced by a model trained to correct the flux errors of non-variable
objects so that their reduced χ² is close to unity. The corrected values may be
useful for light-curve fitting. The ``*_flag`` columns are ``True`` when the
ratio between the original and corrected error falls outside the ``[0.1, 50]``
interval. The method is described in Malanchev et al. (in prep); the source code
is available in the `uncle-val <https://github.com/lincc-frameworks/uncle-val>`_
repository.

.. Placeholder: add the Malanchev et al. (in prep) reference/link once available.

**Corrected magnitude error.** A ``psfMagErr_corrected`` subcolumn is added to
the nested forced-source columns, derived from ``psfFluxErr_corrected`` in the
same way the magnitude errors are derived from the original flux-error columns.

**Visit timestamp.** A ``midpointMjdTai`` column is added to both
``objectForcedSource`` and ``diaObjectForcedSource``, joined from the visit table
for the corresponding visit in the original catalog. That visit table is itself
published with the release; see the :ref:`visit–detector table
<edp2-visit-detector>` below.

**Downcasting.** Some ``float64`` columns are downcast to ``float32`` (see the
warning above and the per-collection lists below).


object_collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Magnitude columns added:

* Object (base level): ``{band}_psfMag``, ``{band}_kronMag``, and
  ``{band}_cModelMag`` for the *ugrizy* bands, plus the corresponding error
  columns.
* ``objectForcedSource``: ``psfMag`` plus the corresponding error column.

Downcast columns (``float64`` → ``float32``):

* Base level (22 columns)::

      exponential_dec  exponential_ra
      g_dec  g_epoch  g_ra
      i_dec  i_epoch  i_ra
      r_dec  r_epoch  r_ra
      sersic_dec  sersic_ra
      u_dec  u_epoch  u_ra
      y_dec  y_epoch  y_ra
      z_dec  z_epoch  z_ra

* ``objectForcedSource``: none.

The ``{band}_ra`` and ``{band}_dec`` columns are downcast from ``float64`` to
``float32``; see the warning above regarding the resulting precision loss.


dia_object_collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Magnitude columns added:

* ``diaSource``: ``psfMag`` and ``scienceMag``, plus the corresponding error
  columns.
* ``diaObjectForcedSource``: ``psfMag`` plus the corresponding error column.

``diaSource`` rows whose ``diaObjectId`` does not appear in the ``DiaObject``
table are not included in the nested light curves: 8,136,150 of 1,000,825,975
rows (0.81%).

Downcast columns (``float64`` → ``float32``):

* Base level (48 columns)::

      g_psfFluxMax  g_psfFluxMaxSlope  g_psfFluxMean  g_psfFluxMeanErr  g_psfFluxMin  g_psfFluxSigma  g_scienceFluxMean  g_scienceFluxMeanErr
      i_psfFluxMax  i_psfFluxMaxSlope  i_psfFluxMean  i_psfFluxMeanErr  i_psfFluxMin  i_psfFluxSigma  i_scienceFluxMean  i_scienceFluxMeanErr
      r_psfFluxMax  r_psfFluxMaxSlope  r_psfFluxMean  r_psfFluxMeanErr  r_psfFluxMin  r_psfFluxSigma  r_scienceFluxMean  r_scienceFluxMeanErr
      u_psfFluxMax  u_psfFluxMaxSlope  u_psfFluxMean  u_psfFluxMeanErr  u_psfFluxMin  u_psfFluxSigma  u_scienceFluxMean  u_scienceFluxMeanErr
      y_psfFluxMax  y_psfFluxMaxSlope  y_psfFluxMean  y_psfFluxMeanErr  y_psfFluxMin  y_psfFluxSigma  y_scienceFluxMean  y_scienceFluxMeanErr
      z_psfFluxMax  z_psfFluxMaxSlope  z_psfFluxMean  z_psfFluxMeanErr  z_psfFluxMin  z_psfFluxSigma  z_scienceFluxMean  z_scienceFluxMeanErr

* ``diaSource`` (20 columns)::

      dipoleAngle  dipoleChi2  dipoleFluxDiff  dipoleFluxDiffErr  dipoleLength
      dipoleMeanFlux  dipoleMeanFluxErr  extendedness
      ixx  ixxPSF  ixy  ixyPSF  iyy  iyyPSF
      snr  timeProcessedMjdTai  trailAngle  trailDec  trailLength  trailRa

* ``diaObjectForcedSource``: none.


.. _edp2-photoz:

object_photoz
------------------------------------------------------

The ``object_photoz`` catalog provides photometric-redshift estimates for EDP2
objects.

.. Placeholder: describe the photo-z estimator(s), output columns, and any caveats once finalized.


.. _edp2-visit-detector:

Visit–detector metadata table
------------------------------------------------------

Alongside the three catalogs, the EDP2 build publishes an auxiliary
``public-files/visit_detector.parquet`` table — one row per (visit, detector),
5,150,198 rows and 59 columns. This is the visit-level metadata table from which
the ``midpointMjdTai`` column on the forced-source light curves is joined. It is a
plain Parquet file (not a HATS catalog) and can be loaded directly with, for
example, ``pandas`` or ``pyarrow`` for additional per-visit and per-detector
information: observation-time MJDs, PSF and image-quality metrics, photometric
zero points, sky background and noise, effective-time metrics, and WCS residuals.

It is keyed by ``visitId`` and ``detectorId`` (also available as ``ccdVisitId``).
The full column list is::

    detectorId  visitId  physical_filter  band  ra  dec
    pixelScale  zenithDistance  expTime  zeroPoint  psfSigma  skyBg  skyNoise
    astromOffsetMean  astromOffsetStd  nPsfStar
    psfStarDeltaE1Median  psfStarDeltaE2Median  psfStarDeltaE1Scatter  psfStarDeltaE2Scatter
    psfStarDeltaSizeMedian  psfStarDeltaSizeScatter  psfStarScaledDeltaSizeScatter
    psfTraceRadiusDelta  psfApFluxDelta  psfApCorrSigmaScaledDelta  maxDistToNearestPsf
    starEMedian  starUnNormalizedEMedian
    effTime  effTimePsfSigmaScale  effTimeSkyBgScale  effTimeZeroPointScale  magLim
    decl  ccdVisitId  detector  seeing  skyRotation
    expMidpt  expMidptMJD  obsStart  obsStartMJD  darkTime
    xSize  ySize
    llcra  llcdec  ulcra  ulcdec  urcra  urcdec  lrcra  lrcdec
    wcsCornerMaxOffset  wcsDetectorPointingResidual  wcsVisitPointingResidual
    wcsPreliminaryDetectorPointingResidual  wcsPreliminaryVisitPointingResidual


How to cite
------------------------------------------------------

If you use LSDB or these catalogs in published research, please follow the
citation instructions on the :doc:`citation page </citation>`. If you use the
corrected flux-error columns, please also cite Malanchev et al. (in prep).


Limitations
------------------------------------------------------

* **Position precision.** The downcast ``float32`` ``{band}_ra`` / ``{band}_dec``
  columns can carry a rounding error larger than their reported errors (see the
  warning above); ``{band}_epoch`` is likewise downcast to ``float32``.
* **Unassociated DIA sources.** 0.81% of ``diaSource`` rows (8,136,150 of
  1,000,825,975) are dropped from the nested light curves because their
  ``diaObjectId`` has no match in the ``DiaObject`` table.
* **Data rights.** Access is restricted to Rubin data-rights holders.


How to ask for help
------------------------------------------------------

There are several places to get help, depending on the nature of your question.
For the full list of channels, events, and office hours, see the
:doc:`Contact Us / Getting Help </contact>` page.

**Questions about LSDB / HATS tooling** (loading the catalogs, cross-matching,
performance):

* Open an issue on the `LSDB GitHub repository
  <https://github.com/astronomy-commons/lsdb/issues/new>`_.
* Ask in the LSST Discovery Alliance Slack, channel `#lincc-frameworks-lsdb
  <https://lsstc.slack.com/archives/C04610PQW9F>`_.
* Drop in to the weekly "HATS/LSDB Drop-in" co-working / office hours (Fridays,
  10:00–11:00 AM US Pacific Time).

**Questions about the Rubin data / science** (the underlying DP2 measurements,
data rights, pipelines):

* Ask on the `LSST Community Forum, Data Products category
  <https://community.lsst.org/c/support/data/34>`_ — the best place for
  questions about Rubin data.


Future plans
------------------------------------------------------

*To be completed.*
