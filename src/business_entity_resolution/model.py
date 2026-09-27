import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score

def calculate_entity_macro_f05(predictions_df, threshold=0.5):
    """
    Calculate the macro F0.5 score at the source1_entity_id level.
    """
    df = predictions_df.copy()
    df['prediction'] = (df['probability'] >= threshold).astype(int)
    
    entity_f05_scores = []
    grouped = df.groupby('source1_entity_id')
    
    for _, group in grouped:
        y_true = group['label'].values
        y_pred = group['prediction'].values
        
        # If true is empty and predicted is empty, it's a perfect match for a singleton (F0.5 = 1.0)
        if y_true.sum() == 0 and y_pred.sum() == 0:
            score = 1.0
        else:
            score = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
            
        entity_f05_scores.append(score)
        
    return np.mean(entity_f05_scores) if entity_f05_scores else 0.0

def calculate_entity_macro_f05_from_dicts(y_true_dict, y_pred_dict):
    """
    Calculate the macro F0.5 score at the source1_entity_id level using fast dictionary lookups.
    """
    total_score = 0.0
    n_entities = len(y_true_dict)
    if n_entities == 0:
        return 0.0

    for s1, true_list in y_true_dict.items():
        pred_list = y_pred_dict.get(s1, [])
        
        n_true = len(true_list)
        n_pred = len(pred_list)
        
        if n_true == 0:
            if n_pred == 0:
                total_score += 1.0
            # else score is 0.0
            continue
            
        if n_pred == 0:
            # score is 0.0
            continue
            
        true_set = set(true_list)
        pred_set = set(pred_list)
        
        tp = len(true_set.intersection(pred_set))
        if tp == 0:
            continue
            
        fp = n_pred - tp
        fn = n_true - tp
        
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        
        score = (1.25 * prec * rec) / (0.25 * prec + rec)
        total_score += score
        
    return total_score / n_entities

def evaluate_pairwise_metrics(predictions_df, threshold=0.5):
    """
    Calculate standard pairwise precision, recall, and F0.5
    """
    y_true = predictions_df['label'].values
    y_pred = (predictions_df['probability'] >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f0_5': f05
    }
