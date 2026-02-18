import pandas as pd

def generate_discovery_report(df_agg, final_probs, threshold, output_path):
    print("📊 Generating Enhanced Discovery Report...")
    df_agg['discovery_score'] = final_probs
    df_agg['is_novel_discovery'] = ((df_agg['discovery_score'] >= threshold) & (df_agg['is_target'] == 0)).astype(int)
    
    report = df_agg.sort_values('discovery_score', ascending=False)
    
    # Save to the 'cleaned' folder for history
    report.to_csv(output_path, index=False)
    
    # Save a copy to the ROOT so app.py can load it instantly
    report.to_csv("top_biological_targets.csv", index=False)
    
    novel_count = df_agg['is_novel_discovery'].sum()
    print(f"🚀 SUCCESS! Found {novel_count} novel candidates. Results mirrored to Root directory.")