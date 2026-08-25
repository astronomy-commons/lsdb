"""LSDB-side integration with hats-registry.

Design choices, matching the earlier hats-registry discussion:
  - hats_registry.HatsRegistry.load() is read-only discovery; nothing here
    ever writes to the registry.
  - A catalog's registry identity is read from a dedicated
    `hats_registry_id` field on its HATS properties -- not assumed to equal
    `catalog_name`, since catalog_name is free-text, mutable, and not
    guaranteed globally unique across providers. Ships today via
    TableProperties' extra="allow" with zero hats release required; can be
    promoted to a first-class typed field later without changing this
    module's behavior.
  - Which registry ref (branch/tag/commit) to query is NOT configured
    here -- it's hats_registry.set_default_ref()/get_default_ref(), since
    the ref is a property of the registry data source itself, not
    something LSDB-specific. Call `hats_registry.set_default_ref(...)`
    directly to pin a session to a specific registry snapshot; every
    function here picks that up automatically.
  - Catalogs with no registry ID are a normal, expected state (most
    existing catalogs today), not an error -- show_extensions() on such a
    catalog returns an empty list rather than raising, but a lower-level
    helper is available to distinguish "not registered" from "registered,
    zero extensions" for callers (e.g. a UI) that need to.
  - Mirror detection (for co-located extensions) is done by comparing the
    catalog's actual opened location (`hc_structure.catalog_base_dir`)
    against the registry's recorded `paths` for that catalog's core entry,
    NOT by a dedicated metadata field -- so it needs zero additional
    catalog metadata beyond `hats_registry_id`, at the cost of being only
    as good as the registry's recorded paths matching reality (see
    `_normalize_path` for the (intentionally modest) normalization this
    tolerates).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import hats_registry
from hats_registry import CoreCatalogEntry, ExtensionCatalogEntry, HatsRegistry
from upath import UPath

if TYPE_CHECKING:
    from lsdb.catalog.catalog import Catalog

__all__ = ["find_extensions", "get_registry_id", "load_extension_entry"]

# Process-lifetime cache, keyed by the *resolved* ref (never None) so that
# a mid-session hats_registry.set_default_ref() call correctly busts to a
# different cache entry rather than silently reusing a stale one.
_registry_cache: dict[str, HatsRegistry] = {}


def _get_registry() -> HatsRegistry:
    """Return a cached HatsRegistry for the currently active default ref,
    loading it on first use for that ref.
    """
    ref = hats_registry.get_default_ref()
    if ref not in _registry_cache:
        _registry_cache[ref] = HatsRegistry.load(ref=ref)
    return _registry_cache[ref]


def refresh_registry() -> HatsRegistry:
    """Force a re-fetch of the registry for the currently active default
    ref, bypassing the cache. Useful in a long-running session (e.g. a
    notebook) where new extensions may have been registered since the
    process started.
    """
    ref = hats_registry.get_default_ref()
    _registry_cache[ref] = HatsRegistry.load(ref=ref)
    return _registry_cache[ref]


def get_registry_id(catalog: "Catalog") -> Optional[str]:
    """Read a catalog's registry identity off its HATS properties, if any.

    Returns None for catalogs that were never registered -- this is the
    common case today and is not an error condition.
    """
    catalog_info = catalog.hc_structure.catalog_info
    if catalog_info.model_extra and "hats_registry_id" in catalog_info.model_extra:
        return catalog_info.model_extra["hats_registry_id"]
    # Falls back to a real attribute lookup for the day hats_registry_id
    # is promoted to a first-class typed field on TableProperties.
    return getattr(catalog_info, "hats_registry_id", None)


def _normalize_path(uri: str) -> str:
    """Normalize a location string for comparison purposes.

    A remote or explicitly-scheme'd URI (https://, s3://, file://, ...) is
    already unambiguous and left as-is (modulo a trailing slash). A bare
    local path -- which may be relative to whatever the caller's cwd
    happened to be when the catalog was opened -- is resolved to an
    absolute, canonical form via pathlib, so it compares correctly against
    a registry-side path regardless of how either one was originally
    written. (HatsRegistry.from_directory() performs the matching
    anchor-to-fixture-root step on its side; this is the other half of
    that, applied to whatever path a catalog was actually opened with.)

    Does NOT resolve symlinks differing between two paths that point to the
    same real file via different routes, nor equivalent-but-differently-
    written remote URIs (e.g. an S3 path with vs. without a region-
    qualified host) -- those remain genuinely unmatched, which just means
    detection falls back to the primary mirror rather than raising. Tighten
    this if that fallback rate turns out to matter in practice.
    """
    path = UPath(uri)
    if path.protocol:
        normalized = str(path)
    else:
        normalized = str(Path(uri).resolve())
    return normalized.rstrip("/")


def detect_mirror(catalog: "Catalog", core_entry: "CoreCatalogEntry") -> Optional[str]:
    """Determine which of a core entry's mirror labels corresponds to where
    `catalog` was actually opened from, by comparing its opened location
    against each entry in `core_entry.paths`.

    Returns None if the catalog's opened location doesn't match any
    registered mirror for its core catalog (e.g. it was opened from an
    unregistered local copy, or via a path written differently than the
    registry records) -- callers should treat that the same as "no
    preference," which naturally falls back to the primary mirror.
    """
    opened_path = _normalize_path(str(catalog.hc_structure.catalog_base_dir))
    for mirror_label, mirror_uri in core_entry.paths.items():
        if _normalize_path(mirror_uri) == opened_path:
            return mirror_label
    return None


def find_extensions(catalog: "Catalog") -> list[ExtensionCatalogEntry]:
    """Discover extensions registered against a given catalog.

    Returns an empty list both when the catalog has no registry ID and
    when it has one but no extensions are registered -- callers who need
    to tell these apart should use `get_registry_id()` directly.
    """
    registry_id = get_registry_id(catalog)
    if registry_id is None:
        return []
    return _get_registry().get_extensions(registry_id)


def load_extension_entry(
    extension: Union[str, ExtensionCatalogEntry],
    source_catalog: Optional["Catalog"] = None,
) -> tuple[ExtensionCatalogEntry, str]:
    """Resolve an extension, by catalog_id or by an already-known entry
    object, to its registry entry plus the specific location to load it
    from.

    Parameters
    ----------
    extension : str or ExtensionCatalogEntry
        Either the extension's own catalog_id (looked up in the registry),
        or an ExtensionCatalogEntry already in hand -- e.g. one returned by
        `find_extensions()` -- in which case no registry lookup is needed
        for the entry itself (though the corresponding core entry is still
        fetched for mirror detection, if `source_catalog` is given).
    source_catalog : Catalog, optional
        If given, the returned location is the copy co-located with
        wherever `source_catalog` was actually opened from (matched via
        `detect_mirror`), falling back to the extension's primary location
        if no co-located copy is registered for that mirror -- or if
        `source_catalog` doesn't itself resolve to any known mirror of its
        own core catalog. Without `source_catalog`, always returns the
        primary location.

    Raises KeyError if `extension` is a catalog_id with no matching
    registry entry.
    """
    registry = _get_registry()
    if isinstance(extension, ExtensionCatalogEntry):
        entry = extension
    else:
        entry = registry.get_extension(extension)
        if entry is None:
            raise KeyError(f"No extension registered with catalog_id '{extension}'")

    mirror = None
    if source_catalog is not None:
        core_entry = registry.get_core(entry.extends)
        if core_entry is not None:
            mirror = detect_mirror(source_catalog, core_entry)

    return entry, entry.resolve_path(mirror)
