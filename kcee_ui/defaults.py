"""Default paths + per-slot model identities for the k-CEE attribution browser.

These match the AlphaGenome MPRAMoCon checkpoints used in the parent project
(see eigen-interactions/eigen_steering.py and the EigenMap notebooks).
Override any of them via env vars.
"""
import os

_REPO = "/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra"

DEFAULT_ATTR_FILE = os.environ.get("KCEE_ATTR_FILE", f"{_REPO}/genomic_targets/data/deeplift_attributions.npz")
DEFAULT_EIGEN_PKL = os.environ.get("KCEE_EIGEN_PKL", f"{_REPO}/genomic_targets/data/eigen_analysis.pkl")
DEFAULT_LIBRARY_CSV = os.environ.get("KCEE_LIBRARY_CSV", f"{_REPO}/data/joint_library_combined.csv")

DEFAULT_SLOTS: list[dict] = [
    {
        "cell_type": "K562",
        "model": "K562_v6_do075",
        "key": "attr_K562",
        "pred_key": "predictions_K562",
        "log2fc_col": "K562_log2FC",
        "path": DEFAULT_ATTR_FILE,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/K562/hits.tsv",
    },
    {
        "cell_type": "HepG2",
        "model": "HepG2_v6_do03",
        "key": "attr_HepG2",
        "pred_key": "predictions_HepG2",
        "log2fc_col": "HepG2_log2FC",
        "path": DEFAULT_ATTR_FILE,
        "finemo_tsv": f"{_REPO}/genomic_targets/data/motif/HepG2/hits.tsv",
    },
    {
        "cell_type": "WTC11",
        "model": "WTC11_twostep_v6_do075",
        "key": "attr_WTC11",
        "pred_key": "predictions_WTC11",
        "log2fc_col": "WTC11_log2FC",
        "path": DEFAULT_ATTR_FILE,
        "finemo_tsv": "",
    },
]
