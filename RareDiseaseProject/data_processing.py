"""
Data processing utilities for Rare Disease Drug Repurposing project
"""

from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET


# --------------------------------------------------
# Base paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"

print("data_processing loaded successfully")


# --------------------------------------------------
# Orphanet Product 1 (Diseases)
# --------------------------------------------------
def load_orphanet_xml(path=None):

    if path is None:
        path = DATASET_DIR / "en_product1.xml"

    tree = ET.parse(path)
    root = tree.getroot()

    records = []

    for disorder in root.findall(".//Disorder"):
        records.append(
            {
                "orpha_code": disorder.findtext("OrphaCode"),
                "disease_name": disorder.findtext("Name"),
            }
        )

    return pd.DataFrame(records)


# --------------------------------------------------
# Orphanet Product 6 (Disease–Gene associations)
# --------------------------------------------------
def load_orphanet_genes_xml(path=None):

    if path is None:
        path = DATASET_DIR / "en_product6.xml"

    tree = ET.parse(path)
    root = tree.getroot()

    records = []

    for disorder in root.findall(".//Disorder"):
        orpha_code = disorder.findtext("OrphaCode")

        for gene in disorder.findall(".//Gene"):
            symbol = gene.findtext("Symbol")

            if symbol:
                records.append(
                    {
                        "orpha_code": orpha_code,
                        "gene_symbol": symbol.upper(),
                    }
                )

    return pd.DataFrame(records).drop_duplicates()


# --------------------------------------------------
# DrugBank
# --------------------------------------------------
def load_drugbank_xml(path=None):

    if path is None:
        path = DATASET_DIR / "drugbank.xml"

    tree = ET.parse(path)
    root = tree.getroot()

    ns = {"db": "http://www.drugbank.ca"}

    records = []

    for drug in root.findall("db:drug", ns):
        drug_name = drug.findtext("db:name", namespaces=ns)

        for target in drug.findall(".//db:target", ns):
            gene = target.findtext(".//db:gene-name", namespaces=ns)

            if gene:
                records.append(
                    {
                        "drug_name": drug_name,
                        "gene_symbol": gene.upper(),
                    }
                )

    return pd.DataFrame(records).drop_duplicates()


# --------------------------------------------------
# STRING protein info
# --------------------------------------------------
def load_string_protein_info(path):

    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1],
    )

    df.columns = ["protein_id", "gene_symbol"]

    df["gene_symbol"] = df["gene_symbol"].str.upper()

    return df.dropna().drop_duplicates()


# --------------------------------------------------
# STRING PPI network
# --------------------------------------------------
def load_string_ppi(path, score_threshold=700):

    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python"
    )

    # Rename to consistent internal format
    df.columns = ["protein_id_1", "protein_id_2", "score"]

    df = df[df["score"] >= score_threshold]

    return df


# --------------------------------------------------
# PPI-based gene expansion
# --------------------------------------------------
def expand_genes_via_ppi(
    orphanet_genes_df,
    protein_info_df,
    ppi_df,
    score_threshold=700,
):

    # Merge gene → protein
    genes_with_protein = orphanet_genes_df.merge(
        protein_info_df,
        on="gene_symbol",
        how="left"
    )

    genes_with_protein = genes_with_protein.dropna(subset=["protein_id"])

    expanded_records = []

    for _, row in genes_with_protein.iterrows():
        orpha_code = row["orpha_code"]
        protein_id = row["protein_id"]

        neighbors = ppi_df[
            (ppi_df["protein_id_1"] == protein_id)
            | (ppi_df["protein_id_2"] == protein_id)
        ]

        for _, n in neighbors.iterrows():

            if n["protein_id_1"] == protein_id:
                neighbor_protein = n["protein_id_2"]
            else:
                neighbor_protein = n["protein_id_1"]

            expanded_records.append(
                {
                    "orpha_code": orpha_code,
                    "protein_id": neighbor_protein,
                    "ppi_score": n["score"],
                }
            )

    expanded_df = pd.DataFrame(expanded_records)

    if expanded_df.empty:
        return expanded_df

    # Map back to gene symbols
    expanded_df = expanded_df.merge(
        protein_info_df,
        on="protein_id",
        how="left"
    )

    expanded_df = expanded_df.dropna(subset=["gene_symbol"]).drop_duplicates()

    return expanded_df


# --------------------------------------------------
# Master tables
# --------------------------------------------------
def create_master_table(orphanet_genes_df, drugbank_df, orphanet_df):

    df = orphanet_genes_df.merge(
        drugbank_df,
        on="gene_symbol",
        how="left",
    )

    df = df.merge(
        orphanet_df,
        on="orpha_code",
        how="left",
    )

    return df


def create_master_table_with_ppi(
    orphanet_genes_df,
    drugbank_df,
    orphanet_df,
    expanded_genes_df,
):

    combined_genes = pd.concat(
        [
            orphanet_genes_df[["orpha_code", "gene_symbol"]],
            expanded_genes_df[["orpha_code", "gene_symbol"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    df = combined_genes.merge(
        drugbank_df,
        on="gene_symbol",
        how="left",
    )

    df = df.merge(
        orphanet_df,
        on="orpha_code",
        how="left",
    )

    return df
