---
name: Release Tracker
about: Request release batch of LSDB, HATS, HATS-import packages.
title: 'Release: '
labels: ''
assignees: 'delucchi-cmu'

---


## Packages to release

Please check which packages you would like to be released:

- [ ] hats
- [ ] lsdb
- [ ] hats-import

## Additional context

e.g. major/minor/patch, deadlines, blocking issues, breaking changes, folks to notify

## nested-* updates to include

<!-- DEAR SUBMITTER -- DON'T EDIT BELOW HERE -->

<!-- ================= -->
<!-- PROCESS CHECKLIST -->
<!-- ================= -->

## github and pypi steps

### hats 

- [ ] tag in github
- [ ] confirm on [pypi](https://pypi.org/manage/project/hats/releases/)

### lsdb

- [ ] update pinned versions (e.g. hats and nested) (or confirm no updates to pins)
- [ ] tag in github
- [ ] confirm on [pypi](https://pypi.org/manage/project/lsdb/releases/)

### hats-import

- [ ] update pinned versions (e.g. hats) (or confirm no updates to pins)
- [ ] tag in github
- [ ] confirm on [pypi](https://pypi.org/manage/project/hats-import/releases/)

## conda-forge steps

### hats 

- [ ] request new conda-forge version (open [bot command issue](https://github.com/conda-forge/hats-feedstock/issues/) 
  with title `@conda-forge-admin, please update version`)
- [ ] approve conda-forge PR
- [ ] confirm on [conda-forge](https://anaconda.org/conda-forge/hats)

### lsdb

- [ ] request new conda-forge version (open [bot command issue](https://github.com/conda-forge/lsdb-feedstock/issues/) 
  with title `@conda-forge-admin, please update version`)
- [ ] confirm tagged `hats` and `nested-pandas` versions and approve
- [ ] confirm on [conda-forge](https://anaconda.org/conda-forge/lsdb)

### hats-import

- [ ] request new conda-forge version (open [bot command issue](https://github.com/conda-forge/hats-import-feedstock/issues/)  
  with title `@conda-forge-admin, please update version`)
- [ ] confirm tagged hats version and approve
- [ ] confirm on [conda-forge](https://anaconda.org/conda-forge/hats-import)

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
