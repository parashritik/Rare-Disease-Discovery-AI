import pandas as pd
import joblib
import xml.etree.ElementTree as ET
from pathlib import Path

# Path Configuration
BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "top_biological_targets.csv"
MOD_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "datasets"

class DiscoveryEngine:
    """
    Core engine for the Rare Disease Drug Assistant.
    Integrates XAI (Explainable AI) and GenAI (Narrative Advice).
    """
    def __init__(self):
        # Load pre-computed results from the ML pipeline
        self.results = pd.read_csv(RESULTS_FILE) if RESULTS_FILE.exists() else pd.DataFrame()
        # Build mapping from Orphanet XML data
        self.disease_map = self._build_disease_mapping()
        
        # Load the Random Forest model for feature importance logic
        try:
            self.rf_model = joblib.load(MOD_DIR / "hyper_rf.pkl")
        except Exception:
            self.rf_model = None

    def _build_disease_mapping(self):
        """Parses Orphanet XMLs to map clinical diseases to gene symbols."""
        mapping = {}
        xml_files = [DATA_DIR / "en_product1.xml", DATA_DIR / "en_product6.xml"]
        for path in xml_files:
            if not path.exists(): continue
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                for disorder in root.iter('Disorder'):
                    name_elem = disorder.find('Name')
                    name = name_elem.text if name_elem is not None else "Unknown"
                    for gene in disorder.iter('Gene'):
                        sym_elem = gene.find('Symbol')
                        if sym_elem is not None:
                            symbol = sym_elem.text
                            if name not in mapping: mapping[name] = []
                            if symbol not in mapping[name]: mapping[name].append(symbol)
            except Exception:
                continue
        return mapping

    def get_explanation(self, symbol):
        """
        Bimodal Layer (XAI): Returns mathematical weights for the prediction.
        Focuses on Interaction Skewness and Network Centrality.
        """
        match = self.results[self.results['gene_symbol'].str.upper() == symbol.upper()]
        if match.empty: return {}

        row = match.iloc[0]
        return {
            "Interaction Skewness": round(row.get('ppi_skew', 0), 4),
            "Network Centrality": int(row.get('degree_max', 0)),
            "Functional Connectivity": round(row.get('ppi_mean', 0), 4),
            "Interaction Diversity": round(row.get('ppi_std', 0), 4)
        }

    def format_result(self, row):
        """
        Bimodal Layer (GenAI): Translates scores into 'Assistant Advice'.
        Handles 'No Drug Found' scenarios for De Novo design recommendations.
        """
        raw_drugs = str(row.get('existing_drugs', '0'))
        # Clean check for missing pharmacological data
        has_drugs = raw_drugs not in ['0', '', 'None Identified', 'nan', 'None (Novel Lead)']
        
        score = round(row['discovery_score'], 4)
        is_known = row['is_target'] == 1

        if has_drugs:
            advice = f"Strong candidate for Drug Repurposing. Existing compounds: {raw_drugs}."
        elif score >= 0.90:
            advice = "High-confidence Novel Lead. No existing drugs identified; recommend De Novo Drug Design."
        else:
            advice = "Low-confidence lead. Further biological validation required."

        return {
            "gene_symbol": row['gene_symbol'],
            "discovery_score": score,
            "status": "Known Target" if is_known else "Novel Discovery",
            "existing_drugs": raw_drugs if has_drugs else "No existing drugs identified",
            "assistant_advice": advice,
            "xai_weights": self.get_explanation(row['gene_symbol'])
        }

    def search_by_gene(self, symbol):
        match = self.results[self.results['gene_symbol'].str.upper() == symbol.upper()]
        if match.empty: return {"error": "Gene not found"}
        return self.format_result(match.iloc[0])

    def search_by_disease(self, query):
        matched_disease = next((k for k in self.disease_map.keys() if query.lower() in k.lower()), None)
        if not matched_disease: return {"error": "Disease not found"}
        
        symbols = self.disease_map[matched_disease]
        matches = self.results[self.results['gene_symbol'].isin(symbols)]
        return {
            "disease": matched_disease,
            "results": [self.format_result(r) for _, r in matches.iterrows()]
        }

    def get_top_10_genes(self):
        if self.results.empty: return []
        return [self.format_result(r) for _, r in self.results.head(10).iterrows()]