import pandas as pd
import joblib
import xml.etree.ElementTree as ET
from pathlib import Path

# Setup paths relative to the root folder
BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "top_biological_targets.csv"
MOD_DIR = BASE_DIR / "RareDiseaseProject" / "models"
DATA_DIR = BASE_DIR / "RareDiseaseProject" / "datasets"

class DiscoveryEngine:  # <--- Make sure this line is exactly like this
    def __init__(self):
        # Load the discovery results
        self.results = pd.read_csv(RESULTS_FILE)
        # Load the saved threshold
        self.threshold = joblib.load(MOD_DIR / "hyper_threshold.pkl")
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

    def search_by_gene(self, symbol):
        match = self.results[self.results['gene_symbol'].str.upper() == symbol.upper()]
        return match.to_dict(orient='records')[0] if not match.empty else {"error": "Gene not found"}

    def search_by_disease(self, query):
        matched_disease = next((k for k in self.disease_map.keys() if query.lower() in k.lower()), None)
        if not matched_disease: return {"error": "Disease not found"}
        symbols = self.disease_map[matched_disease]
        matches = self.results[self.results['gene_symbol'].isin(symbols)]
        return {"disease": matched_disease, "genes": matches.to_dict(orient='records')}

    def get_top_10(self):
        return self.results.head(10).to_dict(orient='records')