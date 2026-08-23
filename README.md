# 3D Point Cloud Curve Reconstruction

[![CI](https://github.com/ParsaVictor/point-cloud-curve-reconstruction/actions/workflows/ci.yml/badge.svg)](https://github.com/ParsaVictor/point-cloud-curve-reconstruction/actions)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Benchmark of **five peer-reviewed algorithms** for denoising and curve reconstruction of 3D point clouds from industrial 3D-printing scans. Each algorithm is implemented as a self-contained Jupyter notebook with full parameter search, and the strongest techniques are combined into a reusable Python package.

---

## Pipeline Overview

![Pipeline architecture](results/architecture.png)

The combined pipeline chains four complementary filters — each addressing a different noise regime — followed by a curve-point recovery stage that restores valid measurements removed by aggressive filtering.

---

## Algorithms Benchmarked

| # | Algorithm | Reference | Key idea |
|---|-----------|-----------|----------|
| 1 | **MST curve reconstruction** | Lee (2000) | kNN graph → MST → edge-length pruning → curve tracing |
| 2 | **Parameter grid search** | Liu (2006) | 729-combination exhaustive search over kNN / edge-cut / density hyperparams |
| 3 | **B-spline contour fitting** | Javidrad (2011) | Adaptive layer slicing + geometric control-point extraction for 3D printing |
| 4 | **Feature curve preservation** | Fugacci (2024) | Topologically-aware curve detection to prevent over-smoothing |
| 5 | **Clustering + PCA denoising** | Peng (2022) | LOF + adaptive DBSCAN + PCA curve recovery; basis for the final pipeline |

### References

1. Lee, I.-K. (2000). *Curve reconstruction from unorganized points.* Computer Aided Geometric Design, 17(2), 161–177.
2. Liu, Y.-S. et al. (2006). *Automatic registration of point clouds from 3D scanning.* Computer-Aided Design, 38(11).
3. Javidrad, F. et al. (2011). *Contour curve reconstruction from point cloud data for rapid prototyping.* International Journal of Advanced Manufacturing Technology.
4. Fugacci, U. et al. (2024). *Feature curve extraction from point clouds via persistent homology.* Computer-Aided Design.
5. Peng, X. et al. (2022). *Advanced point cloud denoising combining clustering and PCA.* Pattern Recognition Letters.

---

## Results

Sample outputs on industrial scan data (proprietary — see [Data](#data)):

| File | Input pts | Output pts | Retained |
|------|-----------|------------|----------|
| `1.csv` | ~1 200 | ~980 | ~82 % |
| `6.csv` | ~2 800 | ~2 100 | ~75 % |

Result images (generated from actual pipeline runs):

| Input scan | After reconstruction |
|---|---|
| ![Input](results/1.jpg) | ![Output](results/output.png) |

Additional output samples: [`results/rezult.jpg`](results/rezult.jpg), [`results/2.jpg`](results/2.jpg), [`results/3.jpg`](results/3.jpg).

**Key finding:** Algorithms 3 and 4 produced near-identical outputs on the provided data, suggesting the scan resolution was below the threshold needed to differentiate topologically-aware from geometry-only approaches. The client noted that full scan data was withheld for confidentiality — denser data is expected to reveal a meaningful gap.

---

## Repository Structure

```
point-cloud-curve-reconstruction/
├── notebooks/
│   ├── 01_lee2000_mst_reconstruction.ipynb      # Paper 1
│   ├── 02_liu2006_parameter_explorer.ipynb      # Paper 2 — 729-combo search
│   ├── 03_liu2006_best_result.ipynb             # Paper 2 — best run
│   ├── 04_javidrad2011_contour_curves.ipynb     # Paper 3
│   ├── 05_fugacci2024_feature_curves.ipynb      # Paper 4
│   ├── 06_peng2022_clustering_pca.ipynb         # Paper 5
│   └── 07_final_pipeline.ipynb                 # Combined pipeline
├── src/pointcloud/
│   ├── filters.py    # zscore, kNN, density, LOF filters
│   ├── graph.py      # kNN graph, MST, edge-cut, connected components
│   ├── curves.py     # polynomial curve fitting, curve-point recovery
│   └── pipeline.py   # combined 4-stage pipeline
├── scripts/
│   └── make_architecture_diagram.py
├── tests/            # 43 pytest tests — synthetic data, no GPU
├── data/             # CSV point cloud files
└── results/          # output images and CSVs
```

---

## Installation

```bash
git clone https://github.com/ParsaVictor/point-cloud-curve-reconstruction
cd point-cloud-curve-reconstruction
pip install -r requirements.txt
```

For running the notebooks (requires Open3D, scikit-learn, matplotlib):

```bash
pip install -e ".[notebooks]"
```

---

## Quick Start

```python
import numpy as np
from pointcloud import run_pipeline

pts = np.loadtxt("data/1.csv", delimiter=",")[:, :3]

result = run_pipeline(pts, mad_factor=3.5, preserve_curves=True)
print(f"Input: {result['n_input']} pts  →  Output: {result['n_output']} pts")
# Input: 1194 pts  →  Output: 979 pts

clean = result["clean"]   # (M, 3) numpy array
```

---

## Data

Point cloud CSV files are in `data/` — each row is `x,y,z` (optionally with extra columns ignored by the loader). Files were captured from industrial 3D-printing test pieces; the client retained full-resolution scans for confidentiality, so the provided subset is lower-density than the production dataset.

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

43 tests covering filters, graph construction, curve fitting, and the full pipeline. All tests use synthetic data (helix / arc geometries generated in code) — no external dataset or GPU required.

---

## Reproducing the Architecture Diagram

```bash
python scripts/make_architecture_diagram.py
# → results/architecture.png
```

---

## Related Work

- [pcb-component-classifier](https://github.com/ParsaVictor/pcb-component-classifier) — Random Forest over engineered CV features (93.3 % accuracy)
- [pcb-component-detection-yolov8](https://github.com/ParsaVictor/pcb-component-detection-yolov8) — YOLOv8 baseline with dataset-quality diagnosis

---

## License

MIT © 2026 Mohammad Parsa Karkooti

---

## Author

**Mohammad Parsa Karkooti** — AI & Computer Vision Engineer  
[GitHub](https://github.com/ParsaVictor) · [LinkedIn](https://www.linkedin.com/in/parsa-karkooti) · 1.parsa.karkooti@gmail.com
