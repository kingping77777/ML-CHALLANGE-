import os
import pandas as pd

def generate_fallback():
    print("Generating fallback submission...")
    s1_path = 'dataset/test/test_source1.tsv'
    if not os.path.exists(s1_path):
        print(f"Error: {s1_path} not found.")
        return
        
    print("Loading test_source1.tsv to get all required S1 IDs...")
    df_s1 = pd.read_csv(s1_path, sep='\t', usecols=['entity_id'], dtype=str)
    all_s1_ids = df_s1['entity_id'].tolist()
    
    print(f"Total S1 entities required: {len(all_s1_ids)}")
    
    # Check if we have some existing outputs we can reuse
    existing_matching = {}
    existing_candidates = {}
    
    # Try to load current partial matching results if available (the 500 rows one)
    try:
        # Actually, let's just generate an empty match for everyone
        # to ensure it's fast and guaranteed to pass validation.
        pass
    except Exception as e:
        pass

    print("Creating empty matching_results.tsv...")
    df_matching = pd.DataFrame({
        'source1_entity_id': all_s1_ids,
        'matched_entity_ids': [''] * len(all_s1_ids)
    })
    os.makedirs('fallback_output', exist_ok=True)
    df_matching.to_csv('fallback_output/matching_results.tsv', sep='\t', index=False)
    
    print("Creating empty candidate_pairs.tsv...")
    df_candidates = pd.DataFrame({
        'source1_entity_id': all_s1_ids,
        'candidate_entity_ids': [''] * len(all_s1_ids)
    })
    df_candidates.to_csv('fallback_output/candidate_pairs.tsv', sep='\t', index=False)
    
    print("Fallback files generated in fallback_output/")
    print("Run this to zip it: Compress-Archive -Path fallback_output/matching_results.tsv, fallback_output/candidate_pairs.tsv, code, Documentation_template.md -DestinationPath Fallback_KingPing_submission.zip -Force")

if __name__ == "__main__":
    generate_fallback()
