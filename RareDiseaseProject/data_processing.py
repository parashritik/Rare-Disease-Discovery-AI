import pandas as pd
import xml.etree.ElementTree as ET
import gc
import os

def parse_xml_to_df(path, file_type="orphanet"):
    if path is None or not os.path.exists(path):
        return pd.DataFrame(columns=['gene_symbol', 'is_target'])
    data = []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        if file_type == "orphanet":
            for gene in root.iter('Gene'):
                symbol = gene.find('Symbol')
                if symbol is not None: data.append({'gene_symbol': symbol.text, 'is_target': 1})
        elif file_type == "drugbank":
            for target in root.findall('.//{*}target'):
                for poly in target.findall('.//{*}polypeptide'):
                    sym = poly.find('.//{*}gene-name')
                    if sym is not None: data.append({'gene_symbol': sym.text.upper() if sym.text else None, 'is_target': 1})
        return pd.DataFrame(data).dropna()
    except Exception as e:
        print(f"❌ XML Error: {e}")
        return pd.DataFrame(columns=['gene_symbol', 'is_target'])

def get_optimized_aggregated_data(ppi_path, info_path, p1_path, p6_path, db_path):
    print("🧬 Step 2: Advanced Feature Engineering (Skew & Degree Mean)...")
    
    # Load and Map PPI
    ppi = pd.read_csv(ppi_path, sep=' ')
    info = pd.read_csv(info_path, sep='\t')
    mapping = dict(zip(info['#string_protein_id'], info['preferred_name']))
    ppi['gene_symbol'] = ppi['protein1'].map(mapping)
    
    # Network Degree calculation
    ppi['network_degree'] = ppi.groupby('gene_symbol')['gene_symbol'].transform('count')
    
    # Load Targets
    p1 = parse_xml_to_df(p1_path, "orphanet")
    p6 = parse_xml_to_df(p6_path, "orphanet")
    db = parse_xml_to_df(db_path, "drugbank")
    targets = pd.concat([p1, p6, db]).drop_duplicates(subset=['gene_symbol'])
    
    # Merge
    df = pd.merge(ppi, targets, on='gene_symbol', how='left')
    df['is_target'] = df['is_target'].fillna(0).astype(int)
    
    # ADVANCED AGGREGATION (Matching your Colab logic)
    df_agg = df.groupby('gene_symbol').agg({
        'combined_score': ['mean', 'max', 'std', 'skew', 'count'],
        'network_degree': ['max', 'mean'],
        'is_target': 'max'
    }).reset_index()
    
    df_agg.columns = ['gene_symbol', 'ppi_mean', 'ppi_max', 'ppi_std', 'ppi_skew', 
                      'interaction_count', 'degree_max', 'degree_mean', 'is_target']
    
    del ppi, info, p1, p6, db, targets, df
    gc.collect()
    return df_agg.fillna(0)