"""Default paths + per-slot model identities for the k-CEE attribution browser.

These match the AlphaGenome MPRAMoCon checkpoints used in the parent project
(see eigen-interactions/eigen_steering.py and the EigenMap notebooks).
Override any of them via env vars.
"""
import os

_REPO = "/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra"

_ATTR_NPZ = f"{_REPO}/genomic_targets/data/deeplift_attributions_standardtorch.npz"
_ATTR_H5  = f"{_REPO}/genomic_targets/data/deeplift_attributions_standardtorch.h5"
DEFAULT_ATTR_FILE = os.environ.get("KCEE_ATTR_FILE",
                                    _ATTR_H5 if os.path.exists(_ATTR_H5) else _ATTR_NPZ)

# Older AlphaGenome (eigen-interactions, v6_do0X ckpts) attributions.
_PABLO_NPZ = f"{_REPO}/genomic_targets/data/deeplift_attributions.npz"
_PABLO_H5  = f"{_REPO}/genomic_targets/data/deeplift_attributions.h5"
PABLO_ATTR_FILE = os.environ.get("KCEE_PABLO_ATTR_FILE",
                                  _PABLO_H5 if os.path.exists(_PABLO_H5) else _PABLO_NPZ)

DEFAULT_LIBRARY_CSV = os.environ.get("KCEE_LIBRARY_CSV", f"{_REPO}/data/joint_library_combined.csv")

_LEGNET_DIR = f"{_REPO}/legnet_rep/results"
_LEGNET_HEPG2 = f"{_LEGNET_DIR}/attrs_HepG2.h5"
_LEGNET_K562  = f"{_LEGNET_DIR}/attrs_K562.h5"

# Standardized AlphaGenome encoder ckpts used to produce
# deeplift_attributions_standardtorch.{npz,h5}. The model labels match the
# directory names under /grid/koo/home/shared/models/alphagenome_encoder/torch.
#
# `insert_offset`: position in the 230bp library insert that corresponds to
# attr position 0. Koo lab attrs are saved var-only ((N,4,200) = insert[15:215])
# so insert_offset=15. See .ui-guy/wt_alignment.md.
DEFAULT_SLOTS: list[dict] = [
    {
        "cell_type": "HepG2",
        "model": "mpra_HepG2",
        "key": "attr_HepG2",
        "pred_key": "predictions_HepG2",
        "log2fc_col": "HepG2_log2FC",
        "path": DEFAULT_ATTR_FILE,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/HepG2/hits.tsv",
        "insert_offset": 15,
    },
    {
        "cell_type": "K562",
        "model": "mpra_K562",
        "key": "attr_K562",
        "pred_key": "predictions_K562",
        "log2fc_col": "K562_log2FC",
        "path": DEFAULT_ATTR_FILE,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/K562/hits.tsv",
        "insert_offset": 15,
    },
    {
        "cell_type": "WTC11",
        "model": "mpra_WTC11",
        "key": "attr_WTC11",
        "pred_key": "predictions_WTC11",
        "log2fc_col": "WTC11_log2FC",
        "path": DEFAULT_ATTR_FILE,
        "finemo_tsv": "",
        "insert_offset": 15,
    },
]

# MPRA-LegNet attribution maps: per-cell-line h5, no WTC11.
# Each h5 has datasets `attributions` (N, 4, 200) and `predictions` (N,).
# Var-only saved (insert[15:215]); insert_offset=15.
LEGNET_SLOTS: list[dict] = [
    {
        "cell_type": "HepG2",
        "model": "mpra_legnet_HepG2",
        "key": "attributions",
        "pred_key": "predictions",
        "log2fc_col": "HepG2_log2FC",
        "path": _LEGNET_HEPG2,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/HepG2/hits.tsv",
        "insert_offset": 15,
    },
    {
        "cell_type": "K562",
        "model": "mpra_legnet_K562",
        "key": "attributions",
        "pred_key": "predictions",
        "log2fc_col": "K562_log2FC",
        "path": _LEGNET_K562,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/K562/hits.tsv",
        "insert_offset": 15,
    },
]

# Older AlphaGenome attributions (eigen-interactions pipeline, v6_do0X ckpts).
# Kept selectable for cross-model comparison; UI label is "Pablo models".
# Saved over the full 281bp construct (insert(230)+prom(36)+bar(15)); insert
# starts at construct position 0, so insert_offset=0.
PABLO_SLOTS: list[dict] = [
    {
        "cell_type": "HepG2",
        "model": "pablo_HepG2_v6_do03",
        "key": "attr_HepG2",
        "pred_key": "predictions_HepG2",
        "log2fc_col": "HepG2_log2FC",
        "path": PABLO_ATTR_FILE,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/HepG2/hits.tsv",
        "insert_offset": 0,
    },
    {
        "cell_type": "K562",
        "model": "pablo_K562_v6_do075",
        "key": "attr_K562",
        "pred_key": "predictions_K562",
        "log2fc_col": "K562_log2FC",
        "path": PABLO_ATTR_FILE,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/K562/hits.tsv",
        "insert_offset": 0,
    },
    {
        "cell_type": "WTC11",
        "model": "pablo_WTC11_twostep_v6_do075",
        "key": "attr_WTC11",
        "pred_key": "predictions_WTC11",
        "log2fc_col": "WTC11_log2FC",
        "path": PABLO_ATTR_FILE,
        "finemo_tsv": "",
        "insert_offset": 0,
    },
]

DATA_SOURCES: dict[str, list[dict]] = {
    "Koo lab models": DEFAULT_SLOTS,
    "Pablo models": PABLO_SLOTS,
    "MPRA-LegNet": LEGNET_SLOTS,
}


def infer_insert_offset(attr_L: int) -> int:
    """Map a saved attribution length back to where attr position 0 sits in the
    230bp library insert. Lets the UI Do The Right Thing when an attribution
    file is regenerated with a different layout (e.g. Pablo models switching
    from 281bp full construct to 200bp var-only via attr_shards_uniform/).

    200 -> 15  (var-only, attr starts at insert[15])
    230 -> 0   (bare insert)
    281 -> 0   (insert + prom(36) + bar(15); insert is at construct[0:230])
    other -> 0 (with the explicit slot value overriding via slot.get('insert_offset'))
    """
    if attr_L == 200:
        return 15
    return 0


def slots_for_cell_type(ct: str) -> list[dict]:
    """All slot dicts across data sources whose cell_type matches and whose
    path exists on disk. Used by the "models" k-condition mode."""
    out = []
    for src in (DEFAULT_SLOTS, PABLO_SLOTS, LEGNET_SLOTS):
        for s in src:
            if s.get("cell_type") == ct and s.get("path") and os.path.exists(s["path"]):
                out.append(dict(s))
    return out


def _model_ct_options() -> list[str]:
    """Cell types that appear in >=2 sources (i.e. eligible for cross-model
    comparison)."""
    cts: dict[str, int] = {}
    for src in (DEFAULT_SLOTS, PABLO_SLOTS, LEGNET_SLOTS):
        seen = set()
        for s in src:
            ct = s.get("cell_type")
            if ct and ct not in seen and s.get("path") and os.path.exists(s["path"]):
                cts[ct] = cts.get(ct, 0) + 1
                seen.add(ct)
    return [ct for ct, n in cts.items() if n >= 2]


MODEL_CT_OPTIONS: list[str] = _model_ct_options() or ["HepG2", "K562"]
