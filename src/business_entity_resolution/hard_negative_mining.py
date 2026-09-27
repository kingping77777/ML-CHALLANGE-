import pandas as pd

def select_hard_negatives(train_preds_df, max_per_entity=3, prob_threshold=0.5, total_cap=None):
    """
    Select hard negatives from training predictions.
    Hard negatives are defined as ground truth label = 0 but model probability >= prob_threshold.
    
    Args:
        train_preds_df: DataFrame containing ['source1_entity_id', 'candidate_entity_id', 'probability', 'label']
        max_per_entity: Maximum number of hard negatives to sample per source1_entity_id to ensure diversity
        prob_threshold: Minimum probability to consider a pair a hard negative
        total_cap: (Optional) Absolute max number of hard negatives to return
        
    Returns:
        DataFrame containing the selected hard negative candidate pairs.
    """
    # Filter for false positives with high confidence
    hard_negs = train_preds_df[(train_preds_df['label'] == 0) & (train_preds_df['probability'] >= prob_threshold)].copy()
    
    # Sort by probability descending to get the hardest ones first
    hard_negs = hard_negs.sort_values(by=['source1_entity_id', 'probability'], ascending=[True, False])
    
    # Cap per entity to ensure diversity and prevent a few entities from dominating
    if max_per_entity is not None:
        hard_negs = hard_negs.groupby('source1_entity_id').head(max_per_entity).reset_index(drop=True)
        
    # Global sort by probability
    hard_negs = hard_negs.sort_values(by='probability', ascending=False).reset_index(drop=True)
    
    if total_cap is not None and len(hard_negs) > total_cap:
        hard_negs = hard_negs.head(total_cap)
        
    return hard_negs
