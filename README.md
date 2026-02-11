# Rare Disease Drug Repurposing Assistant

A modular, backend-ready Python application for suggesting drug repurposing candidates for rare diseases using gene–disease and drug–gene associations (Orphanet, DisGeNET, DGIdb/DrugBank). Suitable for VS Code and future deployment (e.g. FastAPI/Flask).

## Features

- **Data loading & cleaning**: Orphanet (rare diseases), DisGeNET (gene–disease), DGIdb/DrugBank (drug–gene); optional STRING PPI. Normalized gene symbols, drug names, and disease IDs.
- **Drug suggestions**:
  - Drugs that target genes associated with a given disease.
  - Drugs for diseases with similar gene associations (Jaccard-style similarity).
- **Modular design**: Easy to wrap in a REST API; functions return structured dicts (JSON-serializable).
- **Optional graph**: `networkx` used to build a disease–gene–drug graph when requested.

## Project structure

```
Capstone/
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # Load and clean datasets; merge associations
│   ├── drug_suggestion.py   # DrugSuggestionEngine and suggestion logic
│   └── utils.py             # Paths, normalization, similarity metrics
├── datasets/                # Place your CSV files here
│   └── README.md            # Dataset file names and column expectations
├── main.py                  # CLI entry point and backend usage example
├── requirements.txt
└── README.md
```

## Setup

1. **Python 3.11+** recommended.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Datasets**: Place your CSVs in `datasets/` (or set paths via environment variables). See **`datasets/README.md`** for:
   - Expected file names: `orphanet.csv`, `disgenet.csv`, `drug_gene.csv`, optional `string_ppi.csv`
   - Expected column names and where to download the data
   - Optional env vars: `ORPHANET_CSV`, `DISGENET_CSV`, `DRUG_GENE_CSV`, `STRING_PPI_CSV`, `DATASETS_DIR`

   If the files are missing, the app runs with small placeholder data so you can test the pipeline.

## Usage

### CLI (test backend)

```bash
# Interactive: prompt for disease name or Orphanet ID
python main.py

# One-shot: disease name or ID
python main.py "Pompe disease"
python main.py ORPHA:558

# Also suggest drugs for similar diseases (by gene-set similarity)
python main.py "Pompe disease" --similar

# Limit number of drugs and output JSON
python main.py ORPHA:558 -k 10 -j
```

### Backend in code (for API or scripts)

```python
from src.data_processing import load_all_datasets, get_merged_associations
from src.drug_suggestion import DrugSuggestionEngine

# Load and merge
data = load_all_datasets()
gene_disease, drug_gene_disease = get_merged_associations(
    data["orphanet"], data["disgenet"], data["drug_gene"]
)
engine = DrugSuggestionEngine(
    gene_disease=gene_disease,
    drug_gene_disease=drug_gene_disease,
    orphanet=data["orphanet"],
    build_graph=False,  # set True if using networkx for graph queries
)

# Structured output (dict / JSON)
result = engine.suggest_drugs_for_disease("Pompe disease", top_k=20)
# result["candidates"] = list of { "drug_name", "gene_count", "genes": [...] }
# result["gene_links"], result["resolved_disease_id"], etc.

result_similar = engine.suggest_drugs_for_similar_diseases(
    "Pompe disease",
    similarity_threshold=0.2,
    top_k_drugs=20,
)
# result_similar["similar_diseases"], result_similar["suggested_drugs"]
```

Returned structures are suitable for `json.dumps()` and for a future FastAPI/Flask API.

## Dataset column hints

Edit `src/data_processing.py` constants if your CSVs use different column names:

- **Orphanet**: `COL_ORPHANET_NAME`, `COL_ORPHANET_ID`
- **DisGeNET**: `COL_DISGENET_GENE`, `COL_DISGENET_DISEASE`, `COL_DISGENET_SCORE`
- **Drug–gene**: `COL_DRUG_GENE_DRUG`, `COL_DRUG_GENE_GENE`, `COL_DRUG_GENE_TYPE`

DisGeNET often uses UMLS CUI for diseases; you can add a mapping to Orphanet IDs in `load_disgenet()` or provide a pre-mapped file.

## Adding a frontend / API

- Use `build_engine()` (or the same logic in `main.py`) once at startup.
- Expose endpoints that call `engine.suggest_drugs_for_disease(...)` and `engine.suggest_drugs_for_similar_diseases(...)` and return the dicts as JSON.
- No changes required inside `data_processing.py`, `drug_suggestion.py`, or `utils.py` for API integration.

## License

Part of your Capstone project; use as needed.
