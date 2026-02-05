HATS catalogs on Hugging Face
========================================================================================

A curated set of LSDB/HATS catalogs is published as a Hugging Face collection:

- `Multimodal Universe with LSDB/HATS on Hugging Face <https://huggingface.co/collections/UniverseTBD/multimodal-universe-lsdb-hats>`__

This is part of our collaboration with Multimodal Universe project. For more details about Multimodal Universe, see `the project page <https://github.com/MultimodalUniverse/MultimodalUniverse>`__.


Requirements
----------------------------------------------------------------------------------------

To load catalogs from Hugging Face via the ``hf://`` URI scheme, install ``huggingface_hub``:

.. code-block:: bash

	pip install huggingface_hub

For more installation options, see the `huggingface_hub documentation <https://github.com/huggingface/huggingface_hub>`__.


Example: load the GAIA dataset
----------------------------------------------------------------------------------------

The Multimodal Universe GAIA dataset is available at:

- `UniverseTBD/mmu_gaia_gaia <https://huggingface.co/datasets/UniverseTBD/mmu_gaia_gaia>`__

You can open it with LSDB using:

.. code-block:: python

	import lsdb

	cat = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_gaia_gaia@main/mmu_gaia_gaia")

