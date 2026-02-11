import pandas as pd

def dataset_statistics(orphanet_df, orphanet_genes_df, master_df):
    print("\n===== DATASET STATISTICS =====")

    # Total diseases
    total_diseases = orphanet_df["orpha_code"].nunique()
    print(f"Total diseases: {total_diseases}")

    # Total genes
    total_genes = orphanet_genes_df["gene_symbol"].nunique()
    print(f"Total unique genes: {total_genes}")

    # Genes per disease
    genes_per_disease = orphanet_genes_df.groupby("orpha_code")["gene_symbol"].nunique()
    print(f"Average genes per disease: {genes_per_disease.mean():.2f}")

    return genes_per_disease


def drug_coverage(master_df):
    print("\n===== DRUG COVERAGE =====")

    # Genes with drugs
    genes_with_drugs = master_df[master_df["drug_name"].notna()]["gene_symbol"].nunique()
    total_genes = master_df["gene_symbol"].nunique()

    print(f"Genes with drugs: {genes_with_drugs}")
    print(f"Genes without drugs: {total_genes - genes_with_drugs}")
    print(f"Drug coverage: {(genes_with_drugs / total_genes) * 100:.2f}%")



def rank_diseases_by_drugs(master_df):
    print("\n===== RANK DISEASES BY NUMBER OF DRUGS =====")

    ranking = (
        master_df.dropna(subset=["drug_name"])
        .groupby(["orpha_code", "disease_name"])["drug_name"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="num_drugs")
    )

    print(ranking.head(10))
    return ranking


def top_diseases_with_no_drugs(master_df):
    print("\n===== TOP DISEASES WITH NO DRUGS =====")

    no_drugs = (
        master_df[master_df["drug_name"].isna()]
        .groupby(["orpha_code", "disease_name"])["gene_symbol"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="num_genes")
    )

    print(no_drugs.head(10))
    return no_drugs


def add_confidence_score(master_df):
    """
    Simple confidence score:
    - More genes linking a disease to a drug → higher confidence
    """
    confidence = (
        master_df.dropna(subset=["drug_name"])
        .groupby(["orpha_code", "disease_name", "drug_name"])["gene_symbol"]
        .nunique()
        .reset_index(name="gene_support")
    )

    # Normalize score
    confidence["confidence_score"] = (
        confidence["gene_support"] / confidence["gene_support"].max()
    )

    print("\n===== DRUG CONFIDENCE SCORES =====")
    print(confidence.head())

    return confidence
