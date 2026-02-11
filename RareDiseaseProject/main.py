"""
Main script for Rare Disease Drug Repurposing project
Includes:
- Data integration
- PPI expansion
- Confidence scoring
- Random Forest Machine Learning
"""

from pathlib import Path

from data_processing import (
    load_orphanet_xml,
    load_orphanet_genes_xml,
    load_drugbank_xml,
    create_master_table,
    create_master_table_with_ppi,
    load_string_protein_info,
    load_string_ppi,
    expand_genes_via_ppi,
)

from analysis import (
    dataset_statistics,
    drug_coverage,
    rank_diseases_by_drugs,
    top_diseases_with_no_drugs,
    add_confidence_score,
)

from ml_model import train_random_forest


# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "cleaned"
MODEL_DIR = BASE_DIR / "models"

STRING_PROTEIN_INFO = DATASET_DIR / "9606.protein.info.v11.5.txt"
STRING_PPI = DATASET_DIR / "9606.protein.links.v11.5.txt"


def main():

    # -------------------------------
    # Step 1: Load datasets
    # -------------------------------
    print("Loading Orphanet Product 1...")
    orphanet_df = load_orphanet_xml()

    print("Loading Orphanet Product 6...")
    orphanet_genes_df = load_orphanet_genes_xml()

    print("Loading DrugBank...")
    drugbank_df = load_drugbank_xml()

    print("Loading STRING protein info...")
    protein_info_df = load_string_protein_info(STRING_PROTEIN_INFO)

    print("Loading STRING PPI network...")
    ppi_df = load_string_ppi(
        STRING_PPI,
        score_threshold=700,
    )

    # -------------------------------
    # Step 2: Expand genes via PPI
    # -------------------------------
    print("Expanding disease genes using PPI...")
    expanded_genes_df = expand_genes_via_ppi(
        orphanet_genes_df,
        protein_info_df,
        ppi_df,
        score_threshold=700,
    )

    # -------------------------------
    # Step 3: Save cleaned CSVs
    # -------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    orphanet_df.to_csv(OUTPUT_DIR / "orphanet_cleaned.csv", index=False)
    orphanet_genes_df.to_csv(
        OUTPUT_DIR / "orphanet_genes_cleaned.csv",
        index=False,
    )
    drugbank_df.to_csv(
        OUTPUT_DIR / "drugbank_cleaned.csv",
        index=False,
    )
    expanded_genes_df.to_csv(
        OUTPUT_DIR / "orphanet_genes_ppi_expanded.csv",
        index=False,
    )

    print("Cleaned and expanded CSVs saved.")

    # -------------------------------
    # Step 4: Create master tables
    # -------------------------------
    print("Creating direct-evidence master table...")
    master_table_direct = create_master_table(
        orphanet_genes_df,
        drugbank_df,
        orphanet_df,
    )

    master_table_direct.to_csv(
        OUTPUT_DIR / "disease_gene_drug_master_direct.csv",
        index=False,
    )

    print("Creating PPI-aware master table...")
    master_table_ppi = create_master_table_with_ppi(
        orphanet_genes_df,
        drugbank_df,
        orphanet_df,
        expanded_genes_df,
    )

    master_table_ppi.to_csv(
        OUTPUT_DIR / "disease_gene_drug_master_with_ppi.csv",
        index=False,
    )

    print("Master tables created successfully.")

    # -------------------------------
    # Step 5: Analysis
    # -------------------------------
    dataset_statistics(orphanet_df, orphanet_genes_df, master_table_ppi)
    drug_coverage(master_table_ppi)

    rank_diseases_by_drugs(master_table_ppi)
    top_diseases_with_no_drugs(master_table_ppi)

    confidence_df = add_confidence_score(master_table_ppi)
    confidence_df.to_csv(
        OUTPUT_DIR / "drug_confidence_scores.csv",
        index=False,
    )

    print("Analysis completed.")

    # -------------------------------
    # Step 6: Machine Learning
    # -------------------------------
    print("\nTraining Random Forest model...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model, feature_importance_df = train_random_forest(
        master_table_ppi,
        MODEL_DIR,
    )

    feature_importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    print("Random Forest model trained and saved successfully.")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
