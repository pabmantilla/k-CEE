# kcee-ui design notes

## 3D eigenvector sphere (v0.16.0)
The 3D eigenvector sphere is a single plotly `Scatter3d` (not the 3-panel
matplotlib log2FC layout): one rotatable + selectable scene reuses the existing
`event.selection → sel_csv` attribution pipeline. Per-CT log2FC is a
color-selector option instead of 3 subplots because Streamlit can't sync or
select across multiple 3D scenes.
