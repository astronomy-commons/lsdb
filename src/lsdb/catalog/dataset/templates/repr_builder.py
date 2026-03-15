from __future__ import annotations
from typing import TYPE_CHECKING

from lsdb.catalog.dataset.templates.catalog_repr import (
    CATALOG_WRAPPER,
    METADATA_SECTION,
    METHODS_SECTION,
    PREVIEW_SECTION,
    SCHEMA_SECTION,
    DOCS_SECTION,
    CELL_ROWS_UNKNOWN,
    DEFAULT_RA_COLUMN,
    DEFAULT_DEC_COLUMN,
)

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog


def build_catalog_html(catalog: "Catalog") -> str:
    """
    Assembles the complete HTML Catalog representation.

    Lazy function, no data loading is triggered
    All information is derived from metadata/schema only

    Formats HTML sections for Jupyter notebook display, not intended for browsers

    Parameters
    ----------
    catalog : Catalog
        The LSDB Catalog instance to represent

    Returns
    -------
    str
        HTML string for Jupyter notebook display
    """

    metadata_html = _build_metadata_section(catalog)
    methods_html = _build_methods_section()
    preview_html = _build_preview_section(catalog)
    schema_html = _build_schema_section(catalog)
    docs_html = _build_docs_section(catalog)

    # Get catalog type
    catalog_type = catalog.hc_structure.catalog_info.catalog_type or "Catalog"

    return CATALOG_WRAPPER.substitute(
        catalog_name=catalog.hc_structure.catalog_info.catalog_name or "Unnamed",
        catalog_type=catalog_type,
        metadata_section=metadata_html,
        methods_section=methods_html,
        preview_section=preview_html,
        schema_section=schema_html,
        docs_section=docs_html,
    )


def _build_metadata_section(catalog: "Catalog") -> str:
    """Builds the metadata section"""
    catalog_info = catalog.hc_structure.catalog_info
    catalog_type = catalog_info.catalog_type or "Standard Catalog"

    # Get row count from metadata (NOT by computing len(catalog))
    total_rows = catalog_info.total_rows
    if total_rows is None:
        total_rows = CELL_ROWS_UNKNOWN
    else:
        total_rows = f"{total_rows:,}"

    # Get column information from schema
    try:
        schema = catalog.hc_structure.schema
        available_columns = len(schema) if schema else 0
    except Exception:
        available_columns = "Unknown"

    loaded_columns = len(catalog.columns)  # metadata

    # Get RA/Dec columns
    ra_column = catalog_info.ra_column or DEFAULT_RA_COLUMN
    dec_column = catalog_info.dec_column or DEFAULT_DEC_COLUMN

    # Check for margin
    margin_status = (
        f'<span class="lsdb-badge">✓ Available</span>'
        if catalog.margin is not None
        else f'<span class="lsdb-badge warning">✗ None</span>'
    )

    return METADATA_SECTION.substitute(
        catalog_name=catalog.hc_structure.catalog_info.catalog_name or "Unnamed",
        catalog_type=catalog_type,
        total_rows=total_rows,
        npartitions=f"{catalog.npartitions:,}",
        loaded_columns=f"{loaded_columns:,}",
        available_columns=f"{available_columns:,}",
        ra_column=ra_column,
        dec_column=dec_column,
        margin_status=margin_status,
    )


def _build_methods_section() -> str:
    """Suggested methods section"""
    methods = [
        {'name': 'catalog.head()', 'desc': 'View first few rows of data'},
        {'name': 'catalog.columns', 'desc': 'List all loaded columns'},
        {'name': 'catalog.dtypes', 'desc': 'View column data types'},
        {'name': 'catalog.plot_pixels()', 'desc': 'Visualize HEALPix pixel distribution'},
        {'name': 'catalog.plot_coverage()', 'desc': 'Show sky coverage map'},
        {'name': 'catalog.get_healpix_pixels()', 'desc': 'Get all HEALPix pixels'},
        {'name': 'catalog.query("expr")', 'desc': 'Filter catalog with query expression'},
        {'name': 'catalog.cone_search(ra, dec, radius)', 'desc': 'Search within a cone region'},
        {'name': 'catalog.estimate_size()', 'desc': 'Estimate catalog memory size'},
        {'name': 'catalog.hc_structure.schema', 'desc': 'View PyArrow schema'},
    ]

    methods_html = ""
    for method in methods:
        methods_html += f"""
        <div class="lsdb-method-item">
            <code>{method['name']}</code>
            <div class="lsdb-method-desc">{method['desc']}</div>
        </div>
        """

    return METHODS_SECTION.substitute(methods_html=methods_html)


def _build_preview_section(catalog: "Catalog") -> str:
    """
    Build the data preview section

    Does NOT call head() or trigger any data loading
    Shows column info from schema
    """
    try:
        schema = catalog.hc_structure.schema
        if schema is None:
            preview_html = "<p class='lsdb-none'>No schema available for preview.</p>"
            preview_status = "No Schema"
        else:
            # Build a lazy preview table from schema only (no data!)
            preview_html = _build_schema_preview_table(schema)
            preview_status = f"{len(schema)} columns in schema"
    except Exception as e:
        preview_html = f"<p class='lsdb-none'>Preview unavailable: {str(e)}</p>"
        preview_status = "Unavailable"

    return PREVIEW_SECTION.substitute(
        preview_html=preview_html,
        preview_status=preview_status,
    )


def _build_schema_preview_table(schema) -> str:
    """Builds HTML table from PyArrow schema"""
    try:
        # Gets field names and types from schema
        fields = []
        for field in schema:
            fields.append((field.name, str(field.type)))

        # Limit to first 10 columns for preview
        fields = fields[:10]

        # Build HTML table
        table_html = '<table class="lsdb-kv"><thead><tr><th>Column</th><th>Type</th></tr></thead><tbody>'
        for name, dtype in fields:
            table_html += f'<tr><td><code>{name}</code></td><td><code>{dtype}</code></td></tr>'
        table_html += '</tbody></table>'

        if len(schema) > 10:
            table_html += f'<p class="lsdb-note">… and {len(schema) - 10} more columns</p>'

        table_html += '<p class="lsdb-note" style="margin-top:8px;">Call <code>catalog.head()</code> to load and preview actual data</p>'

        return table_html
    except Exception:
        return "<p class='lsdb-none'>Could not build schema preview</p>"


def _build_schema_section(catalog: "Catalog") -> str:
    """Builds the PyArrow schema section"""
    try:
        schema = catalog.hc_structure.schema
        if schema is None:
            schema_html = "<span class='lsdb-none'>No schema available</span>"
            schema_status = "None"
        else:
            # Format schema nicely - this is just string formatting, no data loading
            schema_str = str(schema)
            # Escape HTML special characters
            schema_str = (schema_str
                          .replace('&', '&amp;')
                          .replace('<', '&lt;')
                          .replace('>', '&gt;'))
            schema_html = f'<pre>{schema_str}</pre>'
            schema_status = f"{len(schema)} fields"
    except Exception as e:
        schema_html = f"<span class='lsdb-none'>Schema unavailable: {str(e)}</span>"
        schema_status = "Error"

    return SCHEMA_SECTION.substitute(
        schema_html=schema_html,
        schema_status=schema_status,
    )


def _build_docs_section(catalog: "Catalog") -> str:
    """Builds the documentation links section"""
    docs_links = [
        {'text': 'LSDB Documentation', 'url': 'https://docs.lsdb.io', 'title': 'Complete LSDB documentation'},
        {'text': 'Data Catalog', 'url': 'https://data.lsdb.io', 'title': 'Browse available HATS catalogs'},
        {'text': 'API Reference', 'url': 'https://docs.lsdb.io/en/stable/reference/catalog.html',
         'title': 'Catalog API reference'},
        {'text': 'Tutorials', 'url': 'https://docs.lsdb.io/en/stable/tutorials.html',
         'title': 'LSDB tutorials and examples'},
        {'text': 'HATS Format', 'url': 'https://hats.readthedocs.io', 'title': 'HATS catalog format documentation'},
    ]

    docs_links_html = ""
    for link in docs_links:
        docs_links_html += f"""
        <a href="{link['url']}" target="_blank" class="lsdb-docs-link" title="{link['title']}">
            {link['text']}
            <br>
        </a>
        """

    return DOCS_SECTION.substitute(docs_links_html=docs_links_html)

