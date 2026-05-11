---
name: Attribution windows per source
description: Sequence length fed to DeepLIFT and saved-attr coords for each viewer data source
type: project
---

Construct (281bp) = LA(15) + var(200) + RA(15) + prom(36) + bar(15). Variable region = [15:215] within the 230bp insert.

| Source (UI label)   | Generator                          | Model input | Saved attr shape | Saved coords        | Refs shuffle scope |
|---------------------|------------------------------------|-------------|------------------|---------------------|--------------------|
| Koo lab models      | genomic_targets/compute_attr_standardtorch.py | 281bp       | (N, 4, 200)      | var [15:215]        | var only           |
| MPRA-LegNet         | legnet_rep/.../compute_legnet_attrs.py        | 230bp insert| (N, 4, 200)      | var [15:215]        | var only           |
| Pablo models (v6) — pre-fix file `deeplift_attributions.{npz,h5}` | EigenMaps/eigen-interactions/eigen_steering.py (old) | 281bp | (N, 4, 281) | full construct | full construct |
| Pablo models (v6) — uniform file `deeplift_attributions_uniform.{npz,h5}` | EigenMaps/eigen-interactions/eigen_steering.py (2026-05-10) | 281bp | (N, 4, 200) | var [15:215] | var only |

After the 2026-05-10 fix, the **uniform** file matches Koo lab + LegNet exactly on method (hypothetical=True + mean-center), saved shape (N, 4, 200), saved coords (var [15:215]), and reference distribution (var-region-only dinuc shuffle). The pre-fix file is kept on disk but should not be used. Saved as raw hypothetical (NOT input-multiplied); UI computes WT×attr via attr_to_importance.
