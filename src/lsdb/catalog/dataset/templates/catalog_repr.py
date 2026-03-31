from string import Template


CATALOG_WRAPPER = Template(
"""
    <div class="lsdb-catalog">
        <div class="lsdb-header">
            LSDB Catalog — $catalog_name
            <span class="lsdb-badge">$catalog_type</span>
        </div>
        $metadata_section
        $methods_section
        $preview_section
        $schema_section
        $docs_section
    </div>
"""
)


METADATA_SECTION = Template(
"""
    <div class="lsdb-section">
        <div class="lsdb-sec-head">
            <span>📋 Catalog Metadata</span>
            <span class="lsdb-badge">Lazy Loading</span>
        </div>
        <table class="lsdb-kv">
            <tbody>
                <tr>
                    <td>Catalog Name</td>
                    <td>$catalog_name</td>
                </tr>
                <tr>
                    <td>Catalog Type</td>
                    <td>$catalog_type</td>
                </tr>
                <tr>
                    <td>Total Rows</td>
                    <td>$total_rows</td>
                </tr>
                <tr>
                    <td>Number of Partitions</td>
                    <td>$npartitions</td>
                </tr>
                <tr>
                    <td>Columns Loaded</td>
                    <td>$loaded_columns of $available_columns available</td>
                </tr>
                <tr>
                    <td>RA / Dec Columns</td>
                    <td>$ra_column / $dec_column</td>
                </tr>
                <tr>
                    <td>Margin Cache</td>
                    <td>$margin_status</td>
                </tr>
                <tr>
                    <td>Storage Format</td>
                    <td>HATS (Parquet)</td>
                </tr>
            </tbody>
        </table>
    </div>
"""
)


METHODS_SECTION = Template(
"""
    <div class="lsdb-section">
        <div class="lsdb-sec-head methods">
            <span>Suggested Methods</span>
            <span class="lsdb-badge">Quick Access</span>
        </div>
        <div class="lsdb-method-list">
            $methods_html
        </div>
    </div>
"""
)


PREVIEW_SECTION = Template(
    """
        <div class="lsdb-section">
            <div class="lsdb-sec-head">
                <span>Data Preview</span>
                <span class="lsdb-badge">$preview_status</span>
            </div>
            <div class="lsdb-preview-table">
                $preview_html
            </div>
        </div>
    """
)


SCHEMA_SECTION = Template(
    """
        <div class="lsdb-section">
            <div class="lsdb-sec-head schema">
                <span>PyArrow Schema</span>
                <span class="lsdb-badge">$schema_status</span>
            </div>
            <div class="lsdb-schema">
                $schema_html
            </div>
        </div>
    """
)


DOCS_SECTION = Template(
    """
        <div class="lsdb-section">
            <div class="lsdb-sec-head docs">
                <span>Documentation & Resources</span>
            </div>
            <div style="padding: 12px;">
                $docs_links_html
            </div>
        </div>
    """
)

# Constants
CELL_NONE = "Not set"
CELL_MARGIN_NONE = "None"
CELL_ROWS_UNKNOWN = "Unknown"
DEFAULT_RA_COLUMN = "ra"
DEFAULT_DEC_COLUMN = "dec"

