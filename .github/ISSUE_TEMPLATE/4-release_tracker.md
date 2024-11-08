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

<!-- DON'T EDIT BELOW HERE -->



<!-- ================= -->
<!-- PROCESS CHECKLIST -->
<!-- ================= -->

## hats release

- [ ] tag in github
- [ ] confirm on [pypi](https://pypi.org/manage/project/hats/releases/)
- [ ] request new conda-forge version (see [similar issue](https://github.com/conda-forge/hats-feedstock/issues/3)) and approve
- [ ] confirm on [conda-forge](https://anaconda.org/conda-forge/hats)

## lsdb release

- [ ] tag in github
- [ ] confirm on [pypi](https://pypi.org/manage/project/lsdb/releases/)
- [ ] request new conda-forge version (see [similar issue](https://github.com/conda-forge/hats-feedstock/issues/3))
- [ ] confirm tagged hats version and approve
- [ ] confirm on [conda-forge](https://anaconda.org/conda-forge/lsdb)

## hats-import release

- [ ] tag in github
- [ ] confirm on [pypi](https://pypi.org/manage/project/hats-import/releases/)
- [ ] request new conda-forge version (see [similar issue](https://github.com/conda-forge/hats-feedstock/issues/3)) and approve
- [ ] confirm tagged hats version and approve
- [ ] confirm on [conda-forge](https://anaconda.org/conda-forge/hats-import)

## Release announcement

- [ ] send release summary to LSSTC slack #lincc-frameworks-lsdb
