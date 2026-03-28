#
# <div align="center">Predictive Modeling and Explainable AI for Veterinary Safety Profiles, Residue Assessment, and Health Outcomes Using Real-World Data and Physicochemical Properties</div>


<div align="center">
<p>
<a href="#-run-the-full-project">Run the Full Project</a> |
<a href="https://ieeexplore.ieee.org/document/11402599">Paper</a> |
<a href="https://arxiv.org/abs/2510.01520">ArXiv</a>
</p>
</div>


## 🔎 Overview

![Overview of our framework](docs/overview_process.jpg)

## 🗺️ Roadmap
- [x] Release data curation
- [x] Release data preprocessing
- [x] Provide end-to-end runner
- [x] Release ML training processes
- [x] Release model explainability
- [ ] Add LLM few-shot and fine-tuned version

## ⚠️ Important Notes About openFDA Data
**Animal & Veterinary Adverse Events [[/animalandveterinary/event](https://open.fda.gov/apis/animalandveterinary/event/download/)]:**

The Animal & Veterinary Adverse Events dataset obtained from the openFDA Animal & Veterinary Adverse Event Reports endpoint is periodically updated and does not represent a static archive. Historical records may be revised in subsequent releases, and the data are updated on a quarterly basis with an additional reporting delay of approximately 3–6 months from the time reports are received by the FDA to their public release. Consequently, datasets downloaded at different times may not be identical, and analyses based on this source may yield results that differ from those reported in prior studies or publications using earlier versions of the data.

## 🚀 Run The Full Project

This project provides a single orchestrator that runs each stage in order.
See [CVM_FDA/main.py](CVM_FDA/main.py) for the pipeline entry point.

### 1) Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run End-to-End

Default mode runs the classic ML pipeline:

```bash
python3 main.py
```

If you want the LLM path instead:

```bash
python3 main.py --mode llm
```

### 3) Common Options

- Re-run all steps even if outputs already exist:

```bash
python3 main.py --force
```

- Skip downloading OpenFDA dataset (use existing JSONs in data/json_files):

```bash
python3 main.py --skip-download
```

- Skip PubChem enrichment (use existing expanded_cleaned_data.csv and pubchem_compounds.csv):

```bash
python3 main.py --skip-pubchem
```

- Skip modeling altogether:

```bash
python3 main.py --skip-ml
```

### 4) Outputs

The main artifacts are written under the data and results folders:

- Cleaned FDA dataset: [CVM_FDA/data/np_analysis_cleaned.npz](CVM_FDA/data/np_analysis_cleaned.npz)
- PubChem-enriched dataset: [CVM_FDA/data/expanded_cleaned_data.csv](CVM_FDA/data/expanded_cleaned_data.csv)
- PubChem properties: [CVM_FDA/data/pubchem_compounds.csv](CVM_FDA/data/pubchem_compounds.csv)
- SHAP and plots (ML mode): [CVM_FDA/results](CVM_FDA/results)

### Notes

- PubChem and OpenFDA downloading can be slow and network-dependent.
- If you have already downloaded FDA JSONs, keep them in [CVM_FDA/data/json_files](CVM_FDA/data/json_files).

## 📖 Citation

If you find this repository useful, please consider giving a ⭐ or citing our work:

```bibtex
@INPROCEEDINGS{sholehrasa2025predictive,
  author={Sholehrasa, Hossein and Xu, Xuan and Caragea, Doina and Riviere, Jim E. and Jaberi-Douraki, Majid},
  booktitle={2025 IEEE International Conference on Big Data (BigData)}, 
  title={Predictive Modeling and Explainable AI for Veterinary Safety Profiles, Residue Assessment, and Health Outcomes Using Real-World Data and Physicochemical Properties}, 
  year={2025},
  pages={2043-2052},
  doi={10.1109/BigData66926.2025.11402599}
}


