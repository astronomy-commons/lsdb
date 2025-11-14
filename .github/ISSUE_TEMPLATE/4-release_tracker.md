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

*At this point, you may want to skip down to do the hats conda release first, then return here and do the rest--the conda-forge release process can take a little while once triggered.*

### lsdb

- [ ] update pinned versions (e.g. hats and nested) (or confirm no updates to pins)
- [ ] tag in github
- [ ] confirm on [pypi](https://pypi.org/manage/project/lsdb/releases/)

### hats-import

- [ ] update pinned versions (e.g. hats) (or confirm no updates to pins)
- [ ] tag in github
- [ ] confirm on [pypi](https://pypi.org/manage/project/hats-import/releases/)

## conda-forge steps

*Note: if the CI of any of the below steps fails due to a hash mismatch, you need to regenerate the hash used in the `recipe/meta.yaml`. Get the new hash by running `curl -L https://github.com/<org>/<repo>/archive/refs/tags/<tag>.tar.gz | shasum -a 256`, where the url is your desired release's .tar.gz (found in the repo's releases page).*

### hats 

- [ ] request new conda-forge version (open [bot command issue](https://github.com/conda-forge/hats-feedstock/issues/) 
  with title `@conda-forge-admin, please update version`)
- [ ] edit the `recipe/meta.yaml` in the auto-generated PR to match any dependency changes that have been made to the `pyproject.toml` in this release
- [ ] approve conda-forge PR
- [ ] confirm on [conda-forge](https://anaconda.org/conda-forge/hats)

### lsdb

- [ ] request new conda-forge version (open [bot command issue](https://github.com/conda-forge/lsdb-feedstock/issues/) 
  with title `@conda-forge-admin, please update version`)
- [ ] confirm tagged `hats` and `nested-pandas` versions (and any other dependencies that have changed in the pyproject) and approve
- [ ] confirm on [conda-forge](https://anaconda.org/conda-forge/lsdb)

### hats-import

- [ ] request new conda-forge version (open [bot command issue](https://github.com/conda-forge/hats-import-feedstock/issues/)  
  with title `@conda-forge-admin, please update version`)
- [ ] confirm tagged hats version (and any other dependencies that have changed in the pyproject) and approve
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
