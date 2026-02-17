import data_processing, classifiers, analysis, utils
import gc

def main():
    utils.ensure_directories()
    
    # Set your paths here (assuming local datasets folder)
    PATHS = {
        "links": utils.DATA_DIR / '9606.protein.links.v11.5.txt',
        "info":  utils.DATA_DIR / '9606.protein.info.v11.5.txt',
        "db":    utils.DATA_DIR / 'drugbank.xml',
        "p1":    utils.DATA_DIR / 'en_product1.xml',
        "p6":    utils.DATA_DIR / 'en_product6.xml'
    }

    print("⏳ Step 1: Loading & Mapping Data...")
    df_agg = data_processing.get_optimized_aggregated_data(
        PATHS['links'], PATHS['info'], PATHS['p1'], PATHS['p6'], PATHS['db']
    )

    # Training
    opt_threshold, final_probs = classifiers.train_hyper_optimized_model(df_agg, utils.MOD_DIR)

    # Analysis
    analysis.generate_discovery_report(
        df_agg, 
        final_probs, 
        opt_threshold, 
        utils.OUT_DIR / "optimized_capstone_results.csv"
    )

if __name__ == "__main__":
    main()