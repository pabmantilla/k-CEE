"""Default paths + per-slot model identities for the k-CEE attribution browser.

Reads from the canonical attribution tree under `kcee-ui/data/attributions/`,
built by `tools/build_attributions.py`. Every file is (56975, 4, 200) with key
`/attr` + `/predictions`, indexed by `data/attributions/manifest.csv` (which is
joint_library_combined.csv restricted to the 56,975-row intersection set).

Override any path via env vars (`KCEE_ATTR_DIR`, `KCEE_LIBRARY_CSV`, etc.).

Model "families" (Koo lab models / Pablo models / MPRA-LegNet) are the unit of
the data-source picker. Within each family we enumerate (cell_type, method)
slots whose H5 actually exists on disk — the sidebar uses these helpers to
hide unavailable methods/cts rather than disabling them. As soon as a new
attribution file lands (e.g. once the in-flight Pablo+IntGrad and
LegNet+IntGrad sweeps merge), it shows up in the picker without code changes.
"""
import os
import re
from pathlib import Path

# Upstream repo retained for non-attribution artifacts (motif hits, full library
# CSV used by the parent project's notebooks).
_REPO = "/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra"

# Canonical attribution dir bundled inside kcee-ui.
_ATTR_DIR = Path(os.environ.get(
    "KCEE_ATTR_DIR",
    str(Path(__file__).resolve().parents[1] / "data" / "attributions"),
))

_KOO_DIR    = _ATTR_DIR / "koo_standardtorch"
_PABLO_DIR  = _ATTR_DIR / "pablo_ag_ft"
_LEGNET_DIR = _ATTR_DIR / "legnet_ensemble"

# Manifest = the 56,975-row library subset that every canonical attribution
# file is indexed against. The UI defaults to this so positional indexing
# Just Works across all sources. Override via KCEE_LIBRARY_CSV to switch back
# to the upstream 56,980-row library (e.g. for ad-hoc inspection).
DEFAULT_LIBRARY_CSV = os.environ.get(
    "KCEE_LIBRARY_CSV", str(_ATTR_DIR / "manifest.csv"),
)

# Display names for attribution methods (the H5 filename uses the lowercase
# token, the UI shows the pretty form).
METHOD_DISPLAY = {
    "deeplift":  "DeepLIFT",
    "saliency":  "Saliency",
    "intgrad":   "IntGrad",
}


def _koo_slot(ct: str, *, method: str) -> dict:
    return {
        "cell_type": ct,
        "model": f"koo_{method}_{ct}",
        "key": "attr",
        "pred_key": "predictions",
        "log2fc_col": f"{ct}_log2FC",
        "path": str(_KOO_DIR / f"{ct}_{method}.h5"),
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/{ct}/hits.tsv" if ct != "WTC11" else "",
        "insert_offset": 15,
        "method": method,
        "family": "Koo lab models",
    }


def _pablo_slot(ct: str, *, method: str) -> dict:
    return {
        "cell_type": ct,
        "model": f"pablo_ag_ft_{method}_{ct}",
        "key": "attr",
        "pred_key": "predictions",
        "log2fc_col": f"{ct}_log2FC",
        "path": str(_PABLO_DIR / f"{ct}_{method}.h5"),
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/{ct}/hits.tsv" if ct != "WTC11" else "",
        "insert_offset": 15,
        "method": method,
        "family": "Pablo models",
    }


def _legnet_slot(ct: str, *, method: str) -> dict:
    return {
        "cell_type": ct,
        "model": f"legnet_{method}_{ct}",
        "key": "attr",
        "pred_key": "predictions",
        "log2fc_col": f"{ct}_log2FC",
        "path": str(_LEGNET_DIR / f"{ct}_{method}.h5"),
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/{ct}/hits.tsv",
        "insert_offset": 15,
        "method": method,
        "family": "MPRA-LegNet",
    }


# Canonical family registry. Order here is the order in the sidebar picker.
_FAMILY_DEFS: list[tuple[str, callable, list[str], list[str]]] = [
    # (family name, slot factory, cell types, methods)
    ("Koo lab models",  _koo_slot,    ["HepG2", "K562", "WTC11"], ["deeplift", "saliency", "intgrad"]),
    ("Pablo models",    _pablo_slot,  ["HepG2", "K562", "WTC11"], ["deeplift", "saliency", "intgrad"]),
    ("MPRA-LegNet",     _legnet_slot, ["HepG2", "K562", "WTC11"], ["deeplift", "saliency", "intgrad"]),
]


def _build_families() -> dict[str, list[dict]]:
    """Enumerate every (family, ct, method) combination whose H5 exists on disk.

    Order: families in _FAMILY_DEFS order; within a family, cell types then
    methods in declared order."""
    out: dict[str, list[dict]] = {}
    for fam_name, factory, cts, methods in _FAMILY_DEFS:
        slots: list[dict] = []
        for ct in cts:
            for m in methods:
                slot = factory(ct, method=m)
                if slot.get("path") and os.path.exists(slot["path"]):
                    slots.append(slot)
        if slots:
            out[fam_name] = slots
    return out


FAMILIES: dict[str, list[dict]] = _build_families()


# ---- helpers used by the sidebar to build the cascading pickers ----

def family_names() -> list[str]:
    """Families with at least one (ct, method) slot present on disk."""
    return list(FAMILIES.keys())


def family_slug(family: str) -> str:
    """Stable slug for session-state keys."""
    return re.sub(r"[^a-z0-9]+", "_", family.lower()).strip("_")


def methods_for_family(family: str) -> list[str]:
    """Methods (in canonical order) present in any slot of the family."""
    seen: list[str] = []
    for s in FAMILIES.get(family, []):
        m = s.get("method")
        if m and m not in seen:
            seen.append(m)
    return seen


def cts_for_family_method(family: str, method: str) -> list[str]:
    """Cell types for which (family, method) has a slot on disk."""
    return [s["cell_type"] for s in FAMILIES.get(family, []) if s.get("method") == method]


def methods_for_family_ct(family: str, ct: str) -> list[str]:
    """Methods present at (family, ct)."""
    return [s["method"] for s in FAMILIES.get(family, []) if s.get("cell_type") == ct]


def cts_for_family_with_multiple_methods(family: str) -> list[str]:
    """Cell types in `family` that have >=2 methods on disk (eligible for the
    'methods' k-condition mode CT picker). Preserves family declaration order."""
    out: list[str] = []
    for s in FAMILIES.get(family, []):
        ct = s.get("cell_type")
        if ct and ct not in out and len(methods_for_family_ct(family, ct)) >= 2:
            out.append(ct)
    return out


def families_with_multiple_methods_anywhere() -> list[str]:
    """Families that have >=2 methods at >=1 ct (i.e. eligible for the
    'methods' k-condition mode family picker)."""
    return [f for f in family_names() if any(
        len(methods_for_family_ct(f, ct)) >= 2 for ct in
        {s["cell_type"] for s in FAMILIES.get(f, [])}
    )]


def families_for_ct_method(ct: str, method: str) -> list[str]:
    """Families that have a slot at (ct, method) on disk."""
    return [f for f in family_names()
            if any(s.get("cell_type") == ct and s.get("method") == method
                   for s in FAMILIES[f])]


def methods_at_ct_across_families() -> dict[str, list[str]]:
    """Per cell-type, the methods that exist in >=2 families (used to populate
    the method picker in 'models' k-condition mode)."""
    by_ct: dict[str, list[str]] = {}
    all_cts: set[str] = set()
    for slots in FAMILIES.values():
        for s in slots:
            all_cts.add(s["cell_type"])
    for ct in all_cts:
        methods: list[str] = []
        for fam in family_names():
            for m in methods_for_family_ct(fam, ct):
                if m not in methods:
                    methods.append(m)
        # Filter to methods present in >=2 families
        eligible = [m for m in methods if len(families_for_ct_method(ct, m)) >= 2]
        if eligible:
            by_ct[ct] = eligible
    return by_ct


def cts_eligible_for_models_mode() -> list[str]:
    """Cell types that have at least one method present in >=2 families."""
    return list(methods_at_ct_across_families().keys())


def slot_for(family: str, ct: str, method: str) -> dict | None:
    for s in FAMILIES.get(family, []):
        if s.get("cell_type") == ct and s.get("method") == method:
            return dict(s)
    return None


def slots_for_family_method(family: str, method: str) -> list[dict]:
    """All (ct) slots for a (family, method) pair — used by 'cell lines' mode."""
    return [dict(s) for s in FAMILIES.get(family, []) if s.get("method") == method]


def slots_for_ct_method(ct: str, method: str) -> list[dict]:
    """All (family) slots for a (ct, method) pair — used by 'models' mode."""
    out = []
    for fam in family_names():
        s = slot_for(fam, ct, method)
        if s is not None:
            out.append(s)
    return out


def slots_for_family_ct(family: str, ct: str) -> list[dict]:
    """All (method) slots for a (family, ct) pair — used by 'methods' mode."""
    return [dict(s) for s in FAMILIES.get(family, []) if s.get("cell_type") == ct]


def infer_insert_offset(attr_L: int) -> int:
    """Map a saved attribution length back to where attr position 0 sits in the
    230bp library insert. Canonical files are all 200bp var-only (offset 15);
    fallback retained for non-canonical paths set via KCEE_ATTR_DIR.

    200 -> 15  (var-only, attr starts at insert[15])
    230 -> 0   (bare insert)
    281 -> 0   (insert + prom(36) + bar(15); insert is at construct[0:230])
    other -> 0
    """
    if attr_L == 200:
        return 15
    return 0


# ---- backward-compatibility re-exports ----
# Nothing outside app.py imports these (verified via grep), but they are cheap
# and keep one-off scripts / notebooks that might import them working.
DEFAULT_SLOTS: list[dict] = slots_for_family_method("Koo lab models", "deeplift")
KOO_SALIENCY_SLOTS: list[dict] = slots_for_family_method("Koo lab models", "saliency")
KOO_INTGRAD_SLOTS:  list[dict] = slots_for_family_method("Koo lab models", "intgrad")
PABLO_SLOTS:        list[dict] = slots_for_family_method("Pablo models", "deeplift")
LEGNET_SLOTS:       list[dict] = slots_for_family_method("MPRA-LegNet", "deeplift")


def slots_for_cell_type(ct: str) -> list[dict]:
    """Deprecated: superseded by `slots_for_ct_method`. Returns all slots
    (across families and methods) for `ct` whose H5 exists. Kept so external
    callers don't break."""
    out = []
    for fam in family_names():
        for s in FAMILIES[fam]:
            if s.get("cell_type") == ct:
                out.append(dict(s))
    return out


MODEL_CT_OPTIONS: list[str] = cts_eligible_for_models_mode() or ["HepG2", "K562"]
