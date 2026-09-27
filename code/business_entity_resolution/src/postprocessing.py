import pandas as pd
import numpy as np

def build_singleton_features(preds_df):
    """
    Builds entity-level features to train a singleton gate using robust vectorized operations.
    """
    grouped = preds_df.groupby('source1_entity_id')
    
    # Fast aggregations
    stats = grouped['probability'].agg(['max', 'mean', 'median', 'std', 'count']).rename(
        columns={'max': 'max_prob', 'mean': 'mean_prob', 'median': 'median_prob', 'std': 'std_prob', 'count': 'candidate_count'}
    )
    stats['std_prob'] = stats['std_prob'].fillna(0)
    
    # High probability candidate count
    stats['high_prob_count'] = (preds_df['probability'] >= 0.5).groupby(preds_df['source1_entity_id']).sum()
    
    # Top 1 and Top 2 differences
    sorted_df = preds_df.sort_values(by=['source1_entity_id', 'probability'], ascending=[True, False])
    top1 = sorted_df.groupby('source1_entity_id')['probability'].nth(0)
    top2 = sorted_df.groupby('source1_entity_id')['probability'].nth(1)
    
    stats['top1_minus_top2'] = (top1 - top2).reindex(stats.index).fillna(top1)
    stats['top1_minus_mean'] = stats['max_prob'] - stats['mean_prob']
    
    # Target label: 1 if singleton (max label == 0), 0 otherwise
    stats['is_singleton'] = (grouped['label'].max() == 0).astype(int)
    
    return stats.reset_index()


def apply_one_to_one_constraint(preds_df):
    """
    Ensures that each candidate_entity_id is assigned to at most one source1_entity_id.
    Retains the assignment with the highest probability.
    
    Args:
        preds_df: DataFrame containing ['source1_entity_id', 'candidate_entity_id', 'probability']
        
    Returns:
        DataFrame with conflicting assignments removed, plus tracking columns.
    """
    df = preds_df.copy()
    
    # Sort by candidate_entity_id, then probability (descending), then source1_entity_id (deterministic tie-break)
    df = df.sort_values(
        by=['candidate_entity_id', 'probability', 'source1_entity_id'], 
        ascending=[True, False, True]
    )
    
    # Mark duplicates (keep the first, which has the highest prob)
    df['uniqueness_decision'] = ~df.duplicated(subset=['candidate_entity_id'], keep='first')
    
    return df
