# Catalog Description

This catalog is an association catalog that joins the small_sky and small_sky_order1_source catalogs.

This 'soft' version of the association catalog does not contain leaf files.

This catalog was generated using the following snippet:

```python

from hipscat_import.soap import SoapArguments
from hipscat_import.soap.run_soap import run

args = SoapArguments(
    object_catalog_dir="data/small_sky",
    object_id_column="id",
    source_catalog_dir="data/small_sky_order1_source",
    source_object_id_column="obj_id",
    source_id_column="id",
    write_leaf_files=False,
    output_path="data/small_sky_to_o1source",
    output_artifact_name="small_sky_to_o1source",
    overwrite=True,
)

run(args, client)
```

