import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 8: Hard-Negative Mining + XGBoost Model Improvement\n",
    "\n",
    "## 1. Objective\n",
    "The goal is to find difficult negative examples (high-confidence false positives) from the training set and use them to retrain the XGBoost model, making it less overconfident on ambiguous business pairs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import json\n",
    "import os\n",
    "import sys\n",
    "import time\n",
    "import xgboost as xgb\n",
    "\n",
    "# Ensure project root is in sys.path\n",
    "for p in ['.', '..', '../..', 'src']:\n",
    "    abs_p = os.path.abspath(p)\n",
    "    if abs_p not in sys.path:\n",
    "        sys.path.append(abs_p)\n",
    "\n",
    "from src.business_entity_resolution.model import calculate_entity_macro_f05, evaluate_pairwise_metrics\n",
    "from src.business_entity_resolution.hard_negative_mining import select_hard_negatives\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2 & 3. Load Step 7 model & Load training candidate features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "exp_dir = '../experiments' if os.path.exists('../experiments') else 'experiments'\n",
    "models_dir = '../models/xgboost' if os.path.exists('../models/xgboost') else 'models/xgboost'\n",
    "\n",
    "with open(os.path.join(exp_dir, 'feature_schema.json'), 'r') as f:\n",
    "    schema = json.load(f)\n",
    "feature_cols = schema['features']\n",
    "\n",
    "train_df = pd.read_parquet(os.path.join(exp_dir, 'train_pair_features.parquet'))\n",
    "val_df = pd.read_parquet(os.path.join(exp_dir, 'validation_pair_features.parquet'))\n",
    "\n",
    "baseline_model = xgb.Booster()\n",
    "baseline_model.load_model(os.path.join(models_dir, 'entity_matcher.json'))\n",
    "\n",
    "print(f\"Loaded {len(train_df)} training pairs and Step 7 baseline model.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Score training candidates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "dtrain = xgb.DMatrix(train_df[feature_cols])\n",
    "train_preds = baseline_model.predict(dtrain)\n",
    "\n",
    "train_preds_df = train_df[['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'label']].copy()\n",
    "train_preds_df['probability'] = train_preds\n",
    "\n",
    "print(\"Scored training candidates with baseline model.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Analyze high-confidence negatives"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "neg_preds = train_preds_df[train_preds_df['label'] == 0]\n",
    "total_negs = len(neg_preds)\n",
    "\n",
    "thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]\n",
    "print(\"High-Confidence False Positives (Training Set):\")\n",
    "for t in thresholds:\n",
    "    count = (neg_preds['probability'] >= t).sum()\n",
    "    print(f\"P >= {t:.2f}: {count} ({(count/total_negs)*100:.4f}%)\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Define hard-negative sampling & Extract"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# We will use prob_threshold = 0.50 to gather hard negatives, capped at 3 per entity to ensure diversity.\n",
    "hard_negatives_df = select_hard_negatives(train_preds_df, max_per_entity=3, prob_threshold=0.50, total_cap=20000)\n",
    "print(f\"Selected {len(hard_negatives_df)} hard negatives.\")\n",
    "\n",
    "hard_neg_pairs = set(zip(hard_negatives_df['source1_entity_id'], hard_negatives_df['candidate_entity_id']))\n",
    "\n",
    "# Add a flag in train_df to identify hard negatives\n",
    "train_df['is_hard_negative'] = train_df.apply(lambda r: (r['source1_entity_id'], r['candidate_entity_id']) in hard_neg_pairs, axis=1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Build Round 2 training set\n",
    "Strategy: \n",
    "1. Keep all positive pairs\n",
    "2. Keep a representative base set of ordinary negatives (e.g. 10x the positives ~ 50,000)\n",
    "3. Add all selected hard negatives"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "positives = train_df[train_df['label'] == 1]\n",
    "hard_negs_full = train_df[train_df['is_hard_negative'] == True]\n",
    "ordinary_negs = train_df[(train_df['label'] == 0) & (train_df['is_hard_negative'] == False)]\n",
    "\n",
    "np.random.seed(42)\n",
    "sampled_ordinary_negs = ordinary_negs.sample(n=min(50000, len(ordinary_negs)), random_state=42)\n",
    "\n",
    "round2_train_df = pd.concat([positives, sampled_ordinary_negs, hard_negs_full], ignore_index=True)\n",
    "\n",
    "# Shuffle the dataset\n",
    "round2_train_df = round2_train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)\n",
    "\n",
    "print(f\"Round 2 Training Set Size: {len(round2_train_df)}\")\n",
    "print(f\" - Positives: {len(positives)}\")\n",
    "print(f\" - Ordinary Negatives: {len(sampled_ordinary_negs)}\")\n",
    "print(f\" - Hard Negatives: {len(hard_negs_full)}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Retrain XGBoost (Hard Negative Model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train2 = round2_train_df[feature_cols]\n",
    "y_train2 = round2_train_df['label']\n",
    "X_val = val_df[feature_cols]\n",
    "y_val = val_df['label']\n",
    "\n",
    "hn_model = xgb.XGBClassifier(\n",
    "    objective=\"binary:logistic\",\n",
    "    n_estimators=150,  # slightly more estimators for the harder dataset\n",
    "    max_depth=6,\n",
    "    learning_rate=0.1,\n",
    "    subsample=0.8,\n",
    "    colsample_bytree=0.8,\n",
    "    random_state=42,\n",
    "    eval_metric=\"logloss\",\n",
    "    early_stopping_rounds=15\n",
    ")\n",
    "\n",
    "print(\"Training Hard-Negative XGBoost model...\")\n",
    "t0 = time.time()\n",
    "hn_model.fit(X_train2, y_train2, eval_set=[(X_val, y_val)], verbose=False)\n",
    "print(f\"Training complete in {time.time()-t0:.2f} seconds. Best iteration: {hn_model.best_iteration}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Score validation set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "val_preds_hn = hn_model.predict_proba(X_val)[:, 1]\n",
    "val_df_hn = val_df[['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'label']].copy()\n",
    "val_df_hn['probability'] = val_preds_hn\n",
    "\n",
    "# Also score validation using the baseline model for comparison\n",
    "dval = xgb.DMatrix(X_val)\n",
    "val_preds_base = baseline_model.predict(dval)\n",
    "val_df_base = val_df_hn.copy()\n",
    "val_df_base['probability'] = val_preds_base\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10 & 11. Compare baseline vs hard-negative model & Threshold robustness"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]\n",
    "\n",
    "comparison = []\n",
    "for t in thresholds:\n",
    "    base_pm = evaluate_pairwise_metrics(val_df_base, threshold=t)\n",
    "    base_macro = calculate_entity_macro_f05(val_df_base, threshold=t)\n",
    "    \n",
    "    hn_pm = evaluate_pairwise_metrics(val_df_hn, threshold=t)\n",
    "    hn_macro = calculate_entity_macro_f05(val_df_hn, threshold=t)\n",
    "    \n",
    "    comparison.append({\n",
    "        'Threshold': t,\n",
    "        'Base_Prec': base_pm['precision'], 'HN_Prec': hn_pm['precision'],\n",
    "        'Base_Rec': base_pm['recall'], 'HN_Rec': hn_pm['recall'],\n",
    "        'Base_F05': base_pm['f0_5'], 'HN_F05': hn_pm['f0_5'],\n",
    "        'Base_Macro': base_macro, 'HN_Macro': hn_macro\n",
    "    })\n",
    "\n",
    "comp_df = pd.DataFrame(comparison)\n",
    "print(\"Model Comparison across Thresholds:\")\n",
    "print(comp_df[['Threshold', 'Base_Macro', 'HN_Macro', 'Base_Prec', 'HN_Prec', 'Base_Rec', 'HN_Rec']])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 12. Singleton diagnostic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "s1_positive_counts = val_df_hn.groupby('source1_entity_id')['label'].sum()\n",
    "singleton_s1_ids = s1_positive_counts[s1_positive_counts == 0].index\n",
    "singleton_count = len(singleton_s1_ids)\n",
    "\n",
    "def eval_singletons(df, threshold=0.5):\n",
    "    singleton_df = df[df['source1_entity_id'].isin(singleton_s1_ids)]\n",
    "    false_matches = (singleton_df.groupby('source1_entity_id')['probability'].max() >= threshold).sum()\n",
    "    return false_matches\n",
    "\n",
    "print(f\"Total validation singletons: {singleton_count}\")\n",
    "print(f\"Baseline model false matched singletons (T=0.5): {eval_singletons(val_df_base)} ({(eval_singletons(val_df_base)/singleton_count*100):.2f}%)\")\n",
    "print(f\"Hard-Neg model false matched singletons (T=0.5): {eval_singletons(val_df_hn)} ({(eval_singletons(val_df_hn)/singleton_count*100):.2f}%)\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 13. False-positive analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fp_base = val_df_base[(val_df_base['label'] == 0) & (val_df_base['probability'] >= 0.5)]\n",
    "fp_hn = val_df_hn[(val_df_hn['label'] == 0) & (val_df_hn['probability'] >= 0.5)]\n",
    "\n",
    "print(f\"Validation False Positives (T=0.5): Baseline = {len(fp_base)}, Hard-Neg = {len(fp_hn)}\")\n",
    "\n",
    "# Find some that were fixed by HN model\n",
    "fixed_fps = fp_base[~fp_base['candidate_entity_id'].isin(fp_hn['candidate_entity_id'])]\n",
    "print(f\"False Positives fixed by Hard-Neg model: {len(fixed_fps)}\")\n",
    "\n",
    "# Export difficult hard negative examples\n",
    "fp_hn_severe = fp_hn.sort_values(by='probability', ascending=False).head(100)\n",
    "fp_hn_severe.to_csv(os.path.join(exp_dir, 'hard_negative_examples.csv'), index=False)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 14. False-negative analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fn_base = val_df_base[(val_df_base['label'] == 1) & (val_df_base['probability'] < 0.5)]\n",
    "fn_hn = val_df_hn[(val_df_hn['label'] == 1) & (val_df_hn['probability'] < 0.5)]\n",
    "\n",
    "print(f\"Validation False Negatives (T=0.5): Baseline = {len(fn_base)}, Hard-Neg = {len(fn_hn)}\")\n",
    "new_fns = fn_hn[~fn_hn['candidate_entity_id'].isin(fn_base['candidate_entity_id'])]\n",
    "print(f\"New False Negatives introduced by Hard-Neg model: {len(new_fns)}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 15. Score distribution comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Baseline Model (Positive): Mean = {:.4f}, Median = {:.4f}\".format(\n",
    "    val_df_base[val_df_base['label']==1]['probability'].mean(),\n",
    "    val_df_base[val_df_base['label']==1]['probability'].median()))\n",
    "\n",
    "print(\"Hard-Neg Model (Positive): Mean = {:.4f}, Median = {:.4f}\".format(\n",
    "    val_df_hn[val_df_hn['label']==1]['probability'].mean(),\n",
    "    val_df_hn[val_df_hn['label']==1]['probability'].median()))\n",
    "\n",
    "print(\"\\nBaseline Model (Negative): Mean = {:.4f}, Median = {:.4f}\".format(\n",
    "    val_df_base[val_df_base['label']==0]['probability'].mean(),\n",
    "    val_df_base[val_df_base['label']==0]['probability'].median()))\n",
    "\n",
    "print(\"Hard-Neg Model (Negative): Mean = {:.4f}, Median = {:.4f}\".format(\n",
    "    val_df_hn[val_df_hn['label']==0]['probability'].mean(),\n",
    "    val_df_hn[val_df_hn['label']==0]['probability'].median()))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 16. Feature behavior analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "imp_base = baseline_model.get_score(importance_type='gain')\n",
    "imp_hn = hn_model.get_booster().get_score(importance_type='gain')\n",
    "\n",
    "df_imp = pd.DataFrame({'Base_Gain': pd.Series(imp_base), 'HN_Gain': pd.Series(imp_hn)}).fillna(0)\n",
    "df_imp = df_imp.sort_values(by='HN_Gain', ascending=False)\n",
    "print(\"Top Features by Gain (Hard-Negative Model):\")\n",
    "print(df_imp.head(10))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 17 & 18. Model comparison & Selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "comparison_data = [\n",
    "    {\n",
    "        \"Model\": \"Step 7 Baseline\",\n",
    "        \"Training pairs\": len(train_df),\n",
    "        \"Positive pairs\": len(positives),\n",
    "        \"Negative pairs\": len(train_df) - len(positives),\n",
    "        \"Hard negatives added\": 0,\n",
    "        \"Validation pairs\": len(val_df),\n",
    "        \"Pairwise precision\": comp_df.loc[0, 'Base_Prec'],\n",
    "        \"Pairwise recall\": comp_df.loc[0, 'Base_Rec'],\n",
    "        \"Pairwise F0.5\": comp_df.loc[0, 'Base_F05'],\n",
    "        \"Entity macro F0.5\": comp_df.loc[0, 'Base_Macro'],\n",
    "        \"Singleton false-positive rate\": eval_singletons(val_df_base) / singleton_count\n",
    "    },\n",
    "    {\n",
    "        \"Model\": \"Hard Negative\",\n",
    "        \"Training pairs\": len(round2_train_df),\n",
    "        \"Positive pairs\": len(positives),\n",
    "        \"Negative pairs\": len(sampled_ordinary_negs) + len(hard_negs_full),\n",
    "        \"Hard negatives added\": len(hard_negs_full),\n",
    "        \"Validation pairs\": len(val_df),\n",
    "        \"Pairwise precision\": comp_df.loc[0, 'HN_Prec'],\n",
    "        \"Pairwise recall\": comp_df.loc[0, 'HN_Rec'],\n",
    "        \"Pairwise F0.5\": comp_df.loc[0, 'HN_F05'],\n",
    "        \"Entity macro F0.5\": comp_df.loc[0, 'HN_Macro'],\n",
    "        \"Singleton false-positive rate\": eval_singletons(val_df_hn) / singleton_count\n",
    "    }\n",
    "]\n",
    "\n",
    "model_comp_df = pd.DataFrame(comparison_data)\n",
    "model_comp_df.to_csv(os.path.join(exp_dir, 'model_comparison.csv'), index=False)\n",
    "\n",
    "print(\"Model Comparison Summary:\")\n",
    "print(model_comp_df[['Model', 'Entity macro F0.5', 'Pairwise precision', 'Pairwise recall']])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 19. Save artifacts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "hn_model.save_model(os.path.join(models_dir, 'entity_matcher_hard_negative.json'))\n",
    "\n",
    "hard_negatives_df.to_parquet(os.path.join(exp_dir, 'hard_negative_pairs.parquet'), index=False)\n",
    "\n",
    "hn_metrics = {\n",
    "    \"hard_negatives_found\": int((neg_preds['probability'] >= 0.5).sum()),\n",
    "    \"hard_negatives_selected\": len(hard_negatives_df),\n",
    "    \"macro_f05\": float(comp_df.loc[0, 'HN_Macro']),\n",
    "    \"pairwise_precision\": float(comp_df.loc[0, 'HN_Prec']),\n",
    "    \"pairwise_recall\": float(comp_df.loc[0, 'HN_Rec'])\n",
    "}\n",
    "with open(os.path.join(exp_dir, 'hard_negative_metrics.json'), 'w') as f:\n",
    "    json.dump(hn_metrics, f, indent=4)\n",
    "\n",
    "print(\"Saved hard-negative model and artifacts.\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 20. Findings\n",
    "- Mined high-confidence false positives directly from the training candidate pool.\n",
    "- Created a balanced training set (`positives` + `hard negatives` + `sampled easy negatives`).\n",
    "- Retrained an XGBoost model and verified performance directly on the held-out validation set.\n",
    "- Evaluated if precision increased while maintaining acceptable recall."
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open(r'c:\Users\Daksh\Downloads\6ab10eb3b23ba_student_resource\notebooks\08_Hard_Negative_Mining.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)

print("Notebook 08 created successfully.")
