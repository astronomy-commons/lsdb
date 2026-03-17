---
name: Release Tracker
about: Request release batch of LSDB, HATS, HATS-import packages.
title: 'Release: '
labels: ''
assignees: 'delucchi-cmu'

---

## Additional context

e.g. major/minor/patch, deadlines, blocking issues, breaking changes, folks to notify

## nested-* updates to include

<!-- DEAR SUBMITTER -- DON'T EDIT BELOW HERE -->

<!-- ================= -->
<!-- PROCESS CHECKLIST -->
<!-- ================= -->

## github and pypi steps

`hats`, `hats-import`, and `lsdb` are expected to use the same semantic versioning. 
Even if there are no notable changes, please release all three together.

### hats 

- [ ] tag in [github](https://github.com/astronomy-commons/hats/releases)
- [ ] confirm on [pypi](https://pypi.org/project/hats/)

### lsdb

- [ ] update pinned versions (e.g. hats patch release and nested minor release)
- [ ] tag in [github](https://github.com/astronomy-commons/lsdb/releases)
- [ ] confirm on [pypi](https://pypi.org/project/lsdb/)

### hats-import

- [ ] update pinned versions (e.g. hats patch release)
- [ ] tag in [github](https://github.com/astronomy-commons/hats-import/releases)
- [ ] confirm on [pypi](https://pypi.org/project/hats-import/)

## conda-forge steps

### single conda recipe

- [ ] request new conda-forge version (open [bot command issue](https://github.com/conda-forge/lsdb-feedstock/issues/) 
  with title `@conda-forge-admin, please update version`)
- [ ] confirm tagged `hats` and `nested-pandas` versions (and any other dependencies that have changed in the pyproject)
- [ ] approve and merge PR
- [ ] confirm on [conda-forge](https://anaconda.org/conda-forge/lsdb)
- [ ] confirm new version exists in `mamba search lsdb -c conda-forge --override-channels` (or conda search, which is slower)

### tie it together

once releases have been confirmed on conda-forge, confirm that the RSP
environment and notebooks will not be impacted:

- [ ] run the [smoke-test-conda workflow](https://github.com/astronomy-commons/lsdb-rubin/actions/workflows/smoke_test_conda.yml)
  in the `lsdb-rubin` repo.
- [ ] wait until it passes
- [ ] Inspect the `Get dependency changes from installing LSDB` stage and make sure there aren't
  "too many" new packages installed

## Release announcement

- [ ] send release summary to LSSTC slack #lincc-frameworks-lsdb
