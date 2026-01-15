HATS Catalog Structure and Performance
========================================================================================

This page explains how HATS catalogs are laid out on disk and how that structure
influences performance in LSDB. It is a practical summary of the HATS technical
note for users who want to understand how catalogs are organized and why certain
operations are fast or slow.

For the full technical description, see the IVOA HATS note:
https://www.ivoa.net/documents/Notes/HATS/20250822/NOTE-hats-ivoa-1.0-20250822.html

Catalog Layout at a Glance
----------------------------------------------------------------------------------------

A HATS catalog is a directory with:

- a hierarchical spatial partitioning based on HEALPix orders
- Parquet data files for leaf partitions
- optional supplemental tables for cross-matching and indexing
- metadata files that describe the catalog and its partitions

This layout lets LSDB read only the tiles that overlap your query, which is the
main driver of performance.

Catalog Directory Structure
----------------------------------------------------------------------------------------

HATS partitions the sky into a hierarchy of HEALPix tiles. Each tile is mapped to
a directory or file path that encodes its order and pixel index. Leaf tiles hold
the Parquet data files. The directory structure is designed to:

- keep file sizes roughly uniform (adaptive tiling)
- allow fast discovery of spatial coverage
- support parallel reads of independent tiles

Adaptive Tiling
----------------------------------------------------------------------------------------

Unlike a fixed grid, HATS adapts the tile depth to local density. Dense regions
are subdivided more deeply, while sparse regions stay at coarser orders. This
balance keeps partitions at a manageable size and helps avoid hot spots during
queries or cross-matches.

Data Files
----------------------------------------------------------------------------------------

Leaf partitions contain Parquet files with catalog rows. HATS recommends that the
first column is a HEALPix index (often named ``_healpix_29``) to preserve spatial
ordering and enable efficient filtering. Parquet storage provides:

- column pruning (read only what you select)
- predicate pushdown (filter rows without full scans)
- efficient compression for large catalogs

Catalog Collections
----------------------------------------------------------------------------------------

A catalog collection is a grouping of related HATS catalogs, typically a set of complementary datasets. Collections provide a consistent
entry point for discovery and help a user with supplemental tables described below. Collection metadata describes the members and any
shared properties.

Supplemental Tables
----------------------------------------------------------------------------------------

Some catalogs include additional tables to improve performance:

- **Margin cache:** buffers tile boundaries so spatial operations (especially
  cross-matching) do not miss sources near edges. If your dataset is not a catalog collection, you will need to provide a margin cache separately. See :doc:`HATS Margin Caches <hats-margin-caches>` for more details why margin caches are important for cross-matching
- **Index tables:** allows a user quick access given an index. Typical example is finding an object given an Object ID. Without an index table, these lookups are slow because they require FULL dataset scan in order to find a given object. Index Table provides information about the partitions in which given Object ID can be found and therefore minimizie the loading times. 
- **Association tables:** precomputed links between related catalogs to speed
  multi-survey joins.

Skymaps and Coverage Files
----------------------------------------------------------------------------------------

HATS catalogs may include sky coverage maps and other summary assets. These are
used to quickly estimate coverage, data density, or overlap before reading large
tiles. For example, point maps can provide a coarse view of where data exists and help choose spatial filters.

Metadata and Auxiliary Files
----------------------------------------------------------------------------------------

Metadata files describe the catalog and its partitions. Common files include:

- ``properties``: key/value fields describing the catalog and its version
- ``partition_info.csv``: partition list with sizes and spatial info
- ``partition_join_info.csv``: join-aware partition metadata
- ``_metadata`` and ``_common_metadata``: Parquet schema metadata
- ``data_thumbnail.parquet``: small sample of data for quick inspection
- ``collection.properties``: metadata for catalog collections

LSDB uses these files to plan queries, estimate cost, and decide which partitions need to be loaded.

Performance Considerations
----------------------------------------------------------------------------------------


- **Partition count matters:** selecting and operating on larger parts of the sky means that more tiles need to be opened. If possible, use spatial filters to reduce tile selection early.
- **Partition count matters:** random access to many rows which are scattered across the sky will be slow. This is because, even if only one row is needed from a given tile, the entire tile still needs to be downloaded and opened. Therefore, try to design your access patterns to be as spatially coherent as possible. 
- **Column selection is critical:** Parquet column pruning is one of the biggest
  performance wins. Select only what you need. (TODO: HTTP vs S3 column pruning differences?)
- **Metadata scans are not free:** even thought initial catalog access does not load the actual data, it does read the metadata files and can be slow over high-latency links, especially for catalogs with many partitions. The size of metadata scales with the number of partitions, so catalogs with many small partitions will have larger metadata overhead. Local cache should reduce repeated downloads of metadata. 

