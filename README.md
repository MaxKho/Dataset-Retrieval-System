# Information Retrieval Assignment 2

End‑to‑end pipeline for building a **query–document training set**, fine‑tuning a SPECTER‑2 encoder with Adapter Fusion, and ranking documents for ad‑hoc retrieval tasks.  
Everything lives in **src/** – plug‑and‑play from raw data to evaluation reports.

---

## 1 | Set‑up (Conda)

```bash
# create & activate environment (Python 3.10 tested)
conda create -n doc-retrieval python=3.10 -y
conda activate doc-retrieval

# install all Python dependencies
pip install -r requirements.txt

# (optional) expose src/ to Python
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

> **GPU users:** install the matching CUDA‑enabled PyTorch **before** running `pip install`.

---

## 2 | Get the data & model weights 📥

Large files are **not tracked by git**.  
Open `data/README.md` for Google Drive links and SHA‑256 checksums, then:

```bash
cd data
# download the three ZIP archives listed in the README
unzip new-data.zip
unzip precomputed_embeddings.zip
unzip new-weights.zip
```

Your `data/` tree should now match:

```
data/
 ├─ data/1_raw_data/…
 ├─ data/3_final_data/doc_embeddings_*.pkl
 └─ weights/best_finetuned_adhoc_query_adapter_*/
```

---

## 3 | Run something 🚀

### Quick demo (≈30 s)

```bash
python src/demo_retrieval.py        --query "graph neural networks for molecule property prediction"
```

Ranks documents with pre‑computed embeddings & the best adapters.

### Full pipeline (long)

```bash
python src/main_pipeline.py
```

Re‑creates everything from scratch: merges raw data, generates hard negatives, fine‑tunes adapters, computes embeddings, optimises the scoring function, evaluates.

Key flags:

| flag | default | purpose |
|------|---------|---------|
| `--device` | `cpu` | set `cuda:0` for GPU |
| `--config` | `configs/default.yaml` | override stage parameters |
| `--skip-embedding` | `false` | reuse existing embeddings |

---

## 4 | Outputs

* **Embeddings:** `data/data/3_final_data/doc_embeddings_*.pkl`  
* **Fine‑tuned adapters:** `data/weights/*`  
* **Evaluation reports:** `outputs/evaluation/*.json`

---

## 5 | Troubleshooting

| symptom | fix |
|---------|-----|
| `FileNotFoundError: .../1_raw_data/...` | ensure all ZIPs are unzipped into `data/` |
| `OSError: "cuda" not available` | install CUDA‑compatible PyTorch or use `--device cpu` |
| RAM spikes | run `main_pipeline.py` with partial flags to process chunks |

---

## 6 | Licence & citation

Released under **Apache 2.0**.  
If you use this code, please cite the accompanying paper (BibTeX in `CITATION.cff`).

Happy retrieving! 🎉
