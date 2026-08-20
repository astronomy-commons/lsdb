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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import hats_registry
from hats_registry import ExtensionCatalogEntry, HatsRegistry

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


def find_extensions(catalog: "Catalog") -> list[ExtensionCatalogEntry]:
    """Discover extensions registered against a given catalog.

    Returns an empty list both when the catalog has no registry ID and
    when it has one but no extensions are registered -- callers who need
    to tell these apart should use `get_registry_id()` directly.
    """
    registry_id = get_registry_id(catalog)
    if registry_id is None:
        raise ValueError("Catalog has no hats_registry_id; cannot find extensions")
    return _get_registry().get_extensions(registry_id)


def load_extension_entry(extension_id: str) -> ExtensionCatalogEntry:
    """Resolve an extension's registry entry by its own catalog_id.

    Raises KeyError if no such extension is registered. Returns the entry
    (with its `path`) rather than an opened catalog -- actually opening it
    is left to the caller, keeping this module's only dependency on
    `hats_registry`, not a circular import back into lsdb's own loaders.
    """
    entry = _get_registry().get_extension(extension_id)
    if entry is None:
        raise KeyError(f"No extension registered with catalog_id '{extension_id}'")
    return entry
