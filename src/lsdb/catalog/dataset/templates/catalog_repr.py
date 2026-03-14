from string import Template

STYLES = """
<style>
    .lsdb-catalog { 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
        font-size: 16px; 
        color: #222; 
        line-height: 1.4;
    }
    .lsdb-header { 
        font-weight: bold; 
        font-size: 16px; 
        margin-bottom: 12px; 
        color: #1976d2;
        padding: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 6px;
    }
    .lsdb-section { 
        border: 1px solid #ddd; 
        border-radius: 6px;
        overflow: hidden; 
        margin-bottom: 12px; 
        background: white;
    }
    .lsdb-sec-head { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 8px 12px;
        font-weight: bold; 
        font-size: 12px; 
        color: white;
        display: flex; 
        justify-content: space-between;
        align-items: center; 
    }
    .lsdb-sec-head.methods {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .lsdb-sec-head.schema {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    .lsdb-sec-head.docs {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .lsdb-kv { 
        border-collapse: collapse; 
        width: 100%; 
    }
    .lsdb-kv td { 
        padding: 6px 12px; 
        vertical-align: top; 
        border-bottom: 1px solid #f0f0f0;
    }
    .lsdb-kv tr:nth-child(even) { 
        background: #fafafa; 
    }
    .lsdb-kv td:first-child { 
        color: #666; 
        width: 180px;
        font-weight: 500;
    }
    .lsdb-kv td:last-child {
        color: #333;
        font-family: 'Cascadia Code', 'Geist Mono', 'JetBrains Mono';
    }
    .lsdb-none { 
        color: #999; 
        font-style: italic; 
    }
    .lsdb-badge { 
        font-size: 12px; 
        background: #e8f5ee;
        color: #1a7a4a; 
        padding: 2px 10px; 
        border-radius: 12px;
        font-weight: 500;
    }
    .lsdb-badge.warning {
        background: #fff3e0;
        color: #e65100;
    }
    .lsdb-method-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 8px;
        padding: 12px;
    }
    .lsdb-method-item {
        background: #f5f5f5;
        padding: 8px 12px;
        border-radius: 4px;
        border-left: 3px solid #667eea;
    }
    .lsdb-method-item code {
        color: #d81b60;
        font-weight: 500;
        font-family: 'Cascadia Code', 'Geist Mono', 'JetBrains Mono';
    }
    .lsdb-method-desc {
        font-size: 10px;
        color: #666;
        margin-top: 4px;
    }
    .lsdb-schema {
        background: #f8f9fa;
        padding: 12px;
        overflow-x: auto;
        font-size: 10px;
        max-height: 300px;
        overflow-y: auto;
        font-family: 'Cascadia Code', 'Geist Mono', 'JetBrains Mono';
        white-space: pre-wrap;
    }
    .lsdb-docs-link {
        display: inline-block;
        background: #2196f3;
        color: white;
        padding: 6px 12px;
        border-radius: 4px;
        text-decoration: none;
        margin: 4px;
        font-size: 10px;
    }
    .lsdb-docs-link:hover {
        background: #1976d2;
    }
    .lsdb-preview-table {
        overflow-x: auto;
        max-height: 400px;
        overflow-y: auto;
    }
    .lsdb-preview-table table {
        font-size: 10px;
        width: 100%;
    }
    </style>
"""

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

