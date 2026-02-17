import pandas as pd

def generate_discovery_report(df_agg, final_probs, threshold, output_path):
    print("📊 Generating Enhanced Discovery Report...")
    
    # 1. Attach the raw AI scores to all genes
    df_agg['discovery_score'] = final_probs
    
    # 2. Flag the 610 novel candidates
    df_agg['is_novel_discovery'] = ((df_agg['discovery_score'] >= threshold) & (df_agg['is_target'] == 0)).astype(int)
    
    # 3. SAVE EVERYTHING (Required for the Precision-Recall Curve)
    report = df_agg.sort_values('discovery_score', ascending=False)
    report.to_csv(output_path, index=False)
    
    novel_count = df_agg['is_novel_discovery'].sum()
    print(f"🚀 SUCCESS! Found {novel_count} novel candidates. Results saved.")