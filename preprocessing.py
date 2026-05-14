import pandas as pd
import pubchempy as pcp
import time
import random
import json
import os
from tqdm import tqdm

# Set seed for reproducible negative sampling
random.seed(42)

CACHE_FILE = 'smiles_cache.json'
FAILED_FILE = 'failed_drugs.txt'

print("Loading original dataset...")
df_positive = pd.read_csv('db_drug_interactions.csv')

print("Building positive pairs set...")
positive_pairs_set = set(
    df_positive.apply(lambda r: tuple(sorted([str(r['Drug 1']), str(r['Drug 2'])])), axis=1)
)

unique_drugs = pd.concat([df_positive['Drug 1'], df_positive['Drug 2']]).unique()
print(f"Found {len(unique_drugs)} unique drugs.")

# --- PUBCHEM API FETCHING WITH CACHE ---
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        drug_to_smiles = json.load(f)
    print(f"Resumed from cache: {len(drug_to_smiles)} drugs already fetched.")
else:
    drug_to_smiles = {}

if os.path.exists(FAILED_FILE):
    with open(FAILED_FILE, 'r') as f:
        failed_drugs = f.read().splitlines()
else:
    failed_drugs = []

remaining = [d for d in unique_drugs if d not in drug_to_smiles and d not in failed_drugs]

if remaining:
    print(f"Fetching SMILES for {len(remaining)} remaining drugs...")
    for drug in tqdm(remaining, desc="Fetching SMILES"):
        try:
            results = pcp.get_compounds(drug, 'name')
            if results:
                drug_to_smiles[drug] = results[0].isomeric_smiles
            else:
                failed_drugs.append(drug)
            time.sleep(0.5)
        except Exception as e:
            failed_drugs.append(drug)
            time.sleep(1)

        # Save cache every 50 drugs so progress isn't lost on crash
        if len(drug_to_smiles) % 50 == 0:
            with open(CACHE_FILE, 'w') as f:
                json.dump(drug_to_smiles, f)
            with open(FAILED_FILE, 'w') as f:
                f.write('\n'.join(failed_drugs))

# Final save
with open(CACHE_FILE, 'w') as f:
    json.dump(drug_to_smiles, f)
with open(FAILED_FILE, 'w') as f:
    f.write('\n'.join(failed_drugs))

print(f"Fetch complete! Successfully mapped {len(drug_to_smiles)} SMILES. Failed: {len(failed_drugs)}.")

# --- BUILDING THE BALANCED DATASET ---
print("Filtering positive samples...")
mask = df_positive['Drug 1'].isin(drug_to_smiles) & df_positive['Drug 2'].isin(drug_to_smiles)
df_filtered = df_positive[mask].copy()

df_clean_positive = pd.DataFrame({
    'Drug1_Name':   df_filtered['Drug 1'].values,
    'Drug1_SMILES': df_filtered['Drug 1'].map(drug_to_smiles).values,
    'Drug2_Name':   df_filtered['Drug 2'].values,
    'Drug2_SMILES': df_filtered['Drug 2'].map(drug_to_smiles).values,
    'Label': 1
})

num_positive = len(df_clean_positive)
print(f"Usable positive samples: {num_positive}")

print("Generating synthetic negative samples (Class 0)...")
valid_drugs_list = list(drug_to_smiles.keys())
negative_data = []
max_attempts = num_positive * 10
attempts = 0

while len(negative_data) < num_positive and attempts < max_attempts:
    attempts += 1
    d1, d2 = random.sample(valid_drugs_list, 2)
    pair = tuple(sorted([d1, d2]))

    if pair not in positive_pairs_set:
        negative_data.append({
            'Drug1_Name':   d1,
            'Drug1_SMILES': drug_to_smiles[d1],
            'Drug2_Name':   d2,
            'Drug2_SMILES': drug_to_smiles[d2],
            'Label': 0
        })
        positive_pairs_set.add(pair)

df_negative = pd.DataFrame(negative_data)

print("Saving final dataset...")
df_final = pd.concat([df_clean_positive, df_negative]).sample(frac=1, random_state=42).reset_index(drop=True)
df_final.to_csv('drug_interactions_with_smiles.csv', index=False)

print(f"\nSUCCESS! Saved 'drug_interactions_with_smiles.csv'")
print(f"Total rows : {len(df_final):,}  (Class 0: {(df_final['Label']==0).sum():,} | Class 1: {(df_final['Label']==1).sum():,})")
print("You can now run the EDA additions and model_training.ipynb.")
