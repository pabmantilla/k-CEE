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
DEFAULT_SLOTS: list[dict] = [
    {
        "cell_type": "HepG2",
        "model": "mpra_HepG2",
        "key": "attr_HepG2",
        "pred_key": "predictions_HepG2",
        "log2fc_col": "HepG2_log2FC",
        "path": DEFAULT_ATTR_FILE,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/HepG2/hits.tsv",
    },
    {
        "cell_type": "K562",
        "model": "mpra_K562",
        "key": "attr_K562",
        "pred_key": "predictions_K562",
        "log2fc_col": "K562_log2FC",
        "path": DEFAULT_ATTR_FILE,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/K562/hits.tsv",
    },
    {
        "cell_type": "WTC11",
        "model": "mpra_WTC11",
        "key": "attr_WTC11",
        "pred_key": "predictions_WTC11",
        "log2fc_col": "WTC11_log2FC",
        "path": DEFAULT_ATTR_FILE,
        "finemo_tsv": "",
    },
]

# MPRA-LegNet attribution maps: per-cell-line h5, no WTC11.
# Each h5 has datasets `attributions` (N, 4, 230) and `predictions` (N,).
LEGNET_SLOTS: list[dict] = [
    {
        "cell_type": "HepG2",
        "model": "mpra_legnet_HepG2",
        "key": "attributions",
        "pred_key": "predictions",
        "log2fc_col": "HepG2_log2FC",
        "path": _LEGNET_HEPG2,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/HepG2/hits.tsv",
    },
    {
        "cell_type": "K562",
        "model": "mpra_legnet_K562",
        "key": "attributions",
        "pred_key": "predictions",
        "log2fc_col": "K562_log2FC",
        "path": _LEGNET_K562,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/K562/hits.tsv",
    },
]

# Older AlphaGenome attributions (eigen-interactions pipeline, v6_do0X ckpts).
# Kept selectable for cross-model comparison; UI label is "Pablo models".
PABLO_SLOTS: list[dict] = [
    {
        "cell_type": "HepG2",
        "model": "pablo_HepG2_v6_do03",
        "key": "attr_HepG2",
        "pred_key": "predictions_HepG2",
        "log2fc_col": "HepG2_log2FC",
        "path": PABLO_ATTR_FILE,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/HepG2/hits.tsv",
    },
    {
        "cell_type": "K562",
        "model": "pablo_K562_v6_do075",
        "key": "attr_K562",
        "pred_key": "predictions_K562",
        "log2fc_col": "K562_log2FC",
        "path": PABLO_ATTR_FILE,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/K562/hits.tsv",
    },
    {
        "cell_type": "WTC11",
        "model": "pablo_WTC11_twostep_v6_do075",
        "key": "attr_WTC11",
        "pred_key": "predictions_WTC11",
        "log2fc_col": "WTC11_log2FC",
        "path": PABLO_ATTR_FILE,
        "finemo_tsv": "",
    },
]

DATA_SOURCES: dict[str, list[dict]] = {
    "Koo lab models": DEFAULT_SLOTS,
    "Pablo models": PABLO_SLOTS,
    "MPRA-LegNet": LEGNET_SLOTS,
}


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
