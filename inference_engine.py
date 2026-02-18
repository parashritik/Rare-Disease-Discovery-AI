import pandas as pd
import joblib
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "top_biological_targets.csv"
MOD_DIR = BASE_DIR / "RareDiseaseProject" / "models"
DATA_DIR = BASE_DIR / "RareDiseaseProject" / "datasets"

class DiscoveryEngine:
    def __init__(self):
        self.results = pd.read_csv(RESULTS_FILE) if RESULTS_FILE.exists() else pd.DataFrame()
        self.disease_map = self._build_disease_mapping()

    def _build_disease_mapping(self):
        mapping = {}
        xml_files = [DATA_DIR / "en_product1.xml", DATA_DIR / "en_product6.xml"]
        for path in xml_files:
            if not path.exists(): continue
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
                        mapping[name].append(symbol)
        return mapping

    def format_result(self, row):
        # Logic to suggest action
        drugs = row.get('existing_drugs', '0')
        has_drugs = str(drugs) != '0' and drugs != ''
        
        return {
            "gene_symbol": row['gene_symbol'],
            "discovery_score": round(row['discovery_score'], 4),
            "status": "Known Target" if row['is_target'] == 1 else "Novel Discovery",
            "existing_drugs": drugs if has_drugs else "None (Novel Lead)",
            "assistant_advice": "Consider repurposing listed drugs" if has_drugs else "High potential for new drug design"
        }

    def search_by_gene(self, symbol):
        match = self.results[self.results['gene_symbol'].str.upper() == symbol.upper()]
        return self.format_result(match.iloc[0].to_dict()) if not match.empty else {"error": "Gene not found"}

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
        return [self.format_result(r) for _, r in self.results.head(10).iterrows()]