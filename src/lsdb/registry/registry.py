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
from hats_registry.registry import PRIMARY_MIRROR
from upath import UPath

if TYPE_CHECKING:
    import pandas as pd

    from lsdb.catalog.catalog import Catalog

__all__ = [
    "CatalogNotRegisteredError",
    "ExtensionList",
    "NotAnExtensionError",
    "RegistryLookupError",
    "find_extensions",
    "get_registry_id",
    "load_extension_entry",
]


class RegistryLookupError(ValueError):
    """Base class for errors raised by this module's registry-backed
    lookups. Catch this to handle any of them generically, or catch the
    specific subclass below to handle just one case.
    """


class CatalogNotRegisteredError(RegistryLookupError):
    """Raised when an operation needs a catalog's registry identity (e.g.
    discovering or loading its extensions), but the catalog has no
    `hats_registry_id` set -- it was never registered in hats-registry.
    """


class NotAnExtensionError(RegistryLookupError):
    """Raised when an extension passed to `load_extension()` doesn't
    actually extend the catalog it's being loaded against (its registry
    entry's `extends` doesn't match the source catalog's own registry ID).
    """


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


class ExtensionList:
    """Human-readable view over the extensions registered against a
    catalog, as returned by `find_extensions()` / `Catalog.show_extensions()`.

    Supports both positional indexing (`extensions[0]`) and lookup by
    catalog_id (`extensions["bailer-jones"]`) -- both return the underlying
    `ExtensionCatalogEntry`, which `Catalog.load_extension()` accepts
    directly, so `catalog.load_extension(catalog.show_extensions()[0])` (or
    `[...]["some_id"]`) works with no extra step. Also behaves like an
    ordinary read-only sequence (iterable, `len()`, `list(extensions)`).

    Displays as one block per extension (catalog_id heading, then
    key: value fields below it) rather than a table -- a table's fixed
    column width truncates anything sentence-length, which makes it a poor
    fit once an extension carries a free-text `description` (an informal
    field today, via `model_extra` -- see `_description`). Each block shows
    a single resolved `path` -- the copy co-located with wherever the
    source catalog was actually opened from, falling back to the
    extension's primary location -- rather than every registered mirror;
    call `to_frame(show_all_mirrors=True)` for the full picture, or
    `to_frame()` generally if you want a compact, filterable/sortable
    pandas view instead of this default block display.
    """

    def __init__(self, entries: list[ExtensionCatalogEntry], mirror: Optional[str] = None) -> None:
        self._entries = list(entries)
        self._by_id = {e.catalog_id: e for e in self._entries}
        # The source catalog's own detected mirror label (from
        # detect_mirror), if any -- used to decide which path to
        # foreground, both in the default block display and in
        # to_frame()'s default (non-show_all_mirrors) view.
        self._mirror = mirror

    def __getitem__(self, key: Union[int, str, slice]) -> Union[ExtensionCatalogEntry, "ExtensionList"]:
        if isinstance(key, str):
            return self._by_id[key]
        if isinstance(key, slice):
            return ExtensionList(self._entries[key], mirror=self._mirror)
        return self._entries[key]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @staticmethod
    def _description(entry: ExtensionCatalogEntry) -> Optional[str]:
        """Read an optional human-readable description off the entry, if
        present. Not yet a formal schema field -- carried in `model_extra`
        (now preserved since CatalogEntryBase sets extra="allow") until
        promoted to a typed field, same pattern as `hats_registry_id` on
        the HATS side.
        """
        if entry.model_extra and "description" in entry.model_extra:
            return entry.model_extra["description"]
        return getattr(entry, "description", None)

    def _fields_for(self, entry: ExtensionCatalogEntry) -> dict:
        """The key: value fields shown for one extension's block, in
        display order. Shared between the plain-text and HTML renderers so
        they can never drift out of sync with each other.
        """
        used_mirror = self._mirror if self._mirror in entry.paths else PRIMARY_MIRROR

        if "coverage" not in entry.__dict__:
            # optional fields
            entry.coverage = None

        fields = {
            "extends": entry.extends,
            "modality": entry.modality,
            "path": entry.resolve_path(self._mirror),
            "coverage": entry.coverage,
            "mirror": used_mirror,
        }
        description = self._description(entry)
        if description:
            fields["description"] = description
        return fields

    def __repr__(self) -> str:
        if not self._entries:
            return "ExtensionList([])  # no extensions registered"

        blocks = []
        for i, entry in enumerate(self._entries):
            fields = self._fields_for(entry)
            key_width = max(len(k) for k in fields)
            lines = [f"[{i}] {entry.catalog_id}"]
            lines += [f"    {k.ljust(key_width)} : {v}" for k, v in fields.items()]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _repr_html_(self) -> str:
        if not self._entries:
            return "<p><em>No extensions registered.</em></p>"

        parts = []
        for i, entry in enumerate(self._entries):
            fields = self._fields_for(entry)
            rows = "".join(
                f"<tr><td style='padding-right:1em;color:#555;"
                f"vertical-align:top;white-space:nowrap'>{k}</td>"
                f"<td style='white-space:normal'>{v}</td></tr>"
                for k, v in fields.items()
            )
            parts.append(
                f"<div style='margin-bottom:0.75em'>"
                f"<b>[{i}] {entry.catalog_id}</b>"
                f"<table style='margin-top:0.25em'>{rows}</table>"
                f"</div>"
            )
        return "".join(parts)

    def to_frame(self, show_all_mirrors: bool = False) -> "pd.DataFrame":
        """Render as a pandas DataFrame, indexed by catalog_id -- a
        compact, sortable/filterable alternative to the default block
        display, at the cost of truncating any long free-text field (e.g.
        `description`) to fit column width.

        Parameters
        ----------
        show_all_mirrors : bool, default False
            If False (default), shows a single `path` column -- the
            location co-located with wherever the source catalog was
            opened from, or each extension's primary location if there's
            no co-located copy or no source catalog was involved -- plus a
            `mirror` column naming which mirror label that was. If True,
            shows every registered mirror location as its own `path_<label>`
            column instead, across the union of mirror labels present on
            any of the listed extensions.
        """
        import pandas as pd

        if not self._entries:
            columns = (
                ["extends", "modality"] if show_all_mirrors else ["extends", "modality", "mirror", "path"]
            )
            return pd.DataFrame(columns=columns).rename_axis("catalog_id")

        if show_all_mirrors:
            all_labels = sorted({label for e in self._entries for label in e.paths})
            rows = [
                {
                    "extends": e.extends,
                    "modality": e.modality,
                    **{f"path_{label}": e.paths.get(label) for label in all_labels},
                }
                for e in self._entries
            ]
        else:
            rows = [self._fields_for(e) for e in self._entries]

        return pd.DataFrame(rows, index=[e.catalog_id for e in self._entries]).rename_axis("catalog_id")


def find_extensions(catalog: "Catalog") -> ExtensionList:
    """Discover extensions registered against a given catalog.

    Raises CatalogNotRegisteredError if the catalog has no
    `hats_registry_id` -- that's a distinct, actionable situation (the
    catalog was never registered) from "registered, but genuinely has zero
    extensions," which is a normal state and returns an empty
    ExtensionList rather than raising.
    """
    registry_id = get_registry_id(catalog)
    if registry_id is None:
        raise CatalogNotRegisteredError(
            f"Catalog '{catalog.name}' has no hats_registry_id set"
            " so extensions cannot be discovered for it."
        )

    registry = _get_registry()

    # if catalog is an extension itself, throw an error
    if registry.resolve(registry_id).catalog_type == "extension":
        raise TypeError(
            f"Catalog '{catalog.name}' is an extension catalog (catalog_id "
            f"'{registry_id}')"
            ", so it cannot have extensions registered against it."
        )
    entries = registry.get_extensions(registry_id)

    mirror = None
    core_entry = registry.get_core(registry_id)
    if core_entry is not None:
        mirror = detect_mirror(catalog, core_entry)

    return ExtensionList(entries, mirror=mirror)


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
        If given, two things happen beyond simple resolution:

        1. `extension` is validated as actually being one of
           `source_catalog`'s own registered extensions -- i.e. its
           registry entry's `extends` must match `source_catalog`'s own
           `hats_registry_id`. Raises NotAnExtensionError otherwise. This
           requires knowing `source_catalog`'s own registry ID, so also
           raises CatalogNotRegisteredError if `source_catalog` itself has
           none set.
        2. The returned location is the copy co-located with wherever
           `source_catalog` was actually opened from (matched via
           `detect_mirror`), falling back to the extension's primary
           location if no co-located copy is registered for that mirror --
           or if `source_catalog` doesn't itself resolve to any known
           mirror of its own core catalog.

        Without `source_catalog`, neither check happens -- no ownership
        validation, and the primary location is always returned. This is
        an intentionally permissive mode for context-free resolution (e.g.
        "what's the primary path for extension X"), not the path
        `Catalog.load_extension()` takes -- that always passes `self`, so
        both checks always apply there.

    Raises KeyError if `extension` is a catalog_id with no matching
    registry entry. Raises CatalogNotRegisteredError or NotAnExtensionError
    per the `source_catalog` cases above.
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
        registry_id = get_registry_id(source_catalog)
        if registry_id is None:
            raise CatalogNotRegisteredError(
                f"Catalog '{source_catalog.name}' has no hats_registry_id "
                "set -- it isn't registered in hats-registry, so extensions "
                "can't be resolved against it."
            )
        if entry.extends != registry_id:
            raise NotAnExtensionError(
                f"Extension '{entry.catalog_id}' extends '{entry.extends}', "
                f"not '{registry_id}' -- it is not a registered extension "
                "of this catalog."
            )
        core_entry = registry.get_core(registry_id)
        if core_entry is not None:
            mirror = detect_mirror(source_catalog, core_entry)

    return entry, entry.resolve_path(mirror)
