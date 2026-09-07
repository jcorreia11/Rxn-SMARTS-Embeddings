"""Compute exact dataset statistics for Methods section 2.1."""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
from pathlib import Path
import pandas as pd
import numpy as np
from src.rxn_smarts_embeddings.tokenization.smarts_tokenizer import SmartsTokenizer

OUTPUT_JSON = Path('paper/dataset_stats.json')

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/processed/validated_smarts.csv')
df = df[df['valid'].astype(str).str.upper() == 'TRUE'].dropna(subset=['smarts'])
smarts_list = df['smarts'].tolist()

vocab = json.load(open('data/processed/vocab.json'))
tok = SmartsTokenizer()

# ── Character length ───────────────────────────────────────────────────────
char_len = df['smarts'].str.len()
print("=== CHARACTER LENGTH ===")
print(f"N             : {len(df):,}")
print(f"Min           : {char_len.min()}")
print(f"Median        : {char_len.median():.0f}")
print(f"Mean          : {char_len.mean():.1f}")
print(f"Max           : {char_len.max():,}")
print(f"p75           : {char_len.quantile(0.75):.0f}")
print(f"p90           : {char_len.quantile(0.90):.0f}")
print(f"p95           : {char_len.quantile(0.95):.0f}")
print(f"p99           : {char_len.quantile(0.99):.0f}")

# ── Token length ───────────────────────────────────────────────────────────
print("\nComputing token lengths (this may take a minute)...")
token_lengths = [len(tok.tokenize(s)) for s in smarts_list]
tok_len = np.array(token_lengths)

print("\n=== TOKEN LENGTH ===")
print(f"Min           : {tok_len.min()}")
print(f"Median        : {np.median(tok_len):.0f}")
print(f"Mean          : {tok_len.mean():.1f}")
print(f"Max           : {tok_len.max():,}")
print(f"p75           : {np.percentile(tok_len, 75):.0f}")
print(f"p90           : {np.percentile(tok_len, 90):.0f}")
print(f"p95           : {np.percentile(tok_len, 95):.0f}")
print(f"p99           : {np.percentile(tok_len, 99):.0f}")
print(f"≤256 tokens   : {(tok_len <= 256).sum():,} = {(tok_len <= 256).mean()*100:.1f}%")
print(f"≤512 tokens   : {(tok_len <= 512).sum():,} = {(tok_len <= 512).mean()*100:.1f}%")
print(f">512 tokens   : {(tok_len > 512).sum():,} = {(tok_len > 512).mean()*100:.1f}%")

# ── Structural features ────────────────────────────────────────────────────
print("\n=== STRUCTURAL FEATURES ===")
print(f"Atoms/rxn: mean={df['n_atoms'].mean():.1f}, median={df['n_atoms'].median():.0f}, max={df['n_atoms'].max()}")
print(f"Bonds/rxn: mean={df['n_bonds'].mean():.1f}, median={df['n_bonds'].median():.0f}, max={df['n_bonds'].max()}")
print(f"n_reactant_templates distribution:\n{df['n_reactant_templates'].value_counts().sort_index().to_dict()}")
print(f"n_product_templates distribution:\n{df['n_product_templates'].value_counts().sort_index().to_dict()}")

# ── Vocab ──────────────────────────────────────────────────────────────────
print(f"\n=== VOCABULARY ===")
print(f"Vocab size (rule-based): {vocab['size']:,}")

# ── EC number annotations ──────────────────────────────────────────────────
from collections import Counter

RAW_FILES = [
    'data/raw/retrorules-v3.0-metanetx.csv',
    'data/raw/retrorules-v3.0-rhea.csv',
]

frames = []
for p in RAW_FILES:
    raw = pd.read_csv(p, usecols=['TEMPLATE', 'ECS', 'VALID'])
    raw = raw[raw['VALID'].astype(str).str.upper() == 'TRUE']
    frames.append(raw)

raw_all = pd.concat(frames, ignore_index=True)
# Resolve each template's ECS across *all* raw occurrences before deduping —
# a template that appears in both MetaNetX and Rhea can have the annotation
# on only one of the duplicate rows, so deduplicate-then-check (as done
# previously here) silently drops it. This is the same fix documented in
# PAPER_TEMPLATE_JCIM.md §2.1 for the 212,952 vs 214,007 discrepancy.
has_ecs_mask = raw_all['ECS'].notna() & (raw_all['ECS'].str.strip() != '')
ecs_resolved = (
    raw_all[has_ecs_mask]
    .drop_duplicates(subset='TEMPLATE')
    .set_index('TEMPLATE')['ECS']
)
raw = raw_all.drop_duplicates(subset='TEMPLATE').copy()
raw = raw[raw['TEMPLATE'].isin(set(smarts_list))]
raw['ECS'] = raw['TEMPLATE'].map(ecs_resolved)

has_ec = raw['ECS'].notna() & (raw['ECS'].str.strip() != '')
print(f"\n=== EC ANNOTATIONS ===")
print(f"With EC annotation    : {has_ec.sum():,} ({has_ec.mean()*100:.1f}%)")
print(f"Without EC annotation : {(~has_ec).sum():,} ({(~has_ec).mean()*100:.1f}%)")

def ec_labels_at_depth(ecs_str, d):
    if not isinstance(ecs_str, str) or not ecs_str.strip():
        return set()
    labels = set()
    for part in ecs_str.split(';'):
        part = part.strip()
        nums = part.split('.')
        if len(nums) >= d:
            labels.add('.'.join(nums[:d]))
    return labels

ec_stats = {}
ec_names = {'1':'Oxidoreductases','2':'Transferases','3':'Hydrolases',
            '4':'Lyases','5':'Isomerases','6':'Ligases','7':'Translocases'}

for d in [1, 2, 3]:
    col = f'ec_d{d}'
    raw[col] = raw['ECS'].apply(lambda x: ec_labels_at_depth(x, d))
    with_label = raw[col].apply(len) > 0
    n_with = int(with_label.sum())
    flat = [lbl for labels in raw.loc[with_label, col] for lbl in labels]
    cnt = Counter(flat)
    most_cls, most_n = cnt.most_common(1)[0]
    least_cls, least_n = cnt.most_common()[-1]
    print(f"\n  depth={d}: {n_with:,} templates ({n_with/len(raw)*100:.1f}%), "
          f"{len(cnt)} classes | "
          f"most: {most_cls} ({most_n:,}) | least: {least_cls} ({least_n:,})")
    if d == 1:
        print(f"  {'Class':<8} {'Name':<20} {'Templates':>12} {'%':>8}")
        print(f"  {'-'*52}")
        for cls in sorted(cnt.keys()):
            print(f"  EC {cls:<5} {ec_names.get(cls,''):<20} {cnt[cls]:>12,} {cnt[cls]/n_with*100:>7.1f}%")
    ec_stats[f'depth_{d}'] = {
        'n_templates_with_label': n_with,
        'pct_with_label': round(n_with / len(raw) * 100, 1),
        'n_classes': len(cnt),
        'most_populated': {'class': most_cls, 'n': most_n},
        'least_populated': {'class': least_cls, 'n': least_n},
        'distribution': {k: v for k, v in sorted(cnt.items())},
    }

# ── Save JSON ──────────────────────────────────────────────────────────────
results = {
    'n_total': len(df),
    'raw_sources': {
        'metanetx': 302879,
        'rhea': 125842,
        'combined': 428721,
        'duplicates_removed': 66970,
    },
    'character_length': {
        'min': int(char_len.min()),
        'p5':  int(char_len.quantile(0.05)),
        'p25': int(char_len.quantile(0.25)),
        'median': int(char_len.median()),
        'mean': round(float(char_len.mean()), 1),
        'p75': int(char_len.quantile(0.75)),
        'p90': int(char_len.quantile(0.90)),
        'p95': int(char_len.quantile(0.95)),
        'p99': int(char_len.quantile(0.99)),
        'max': int(char_len.max()),
    },
    'token_length_rule_based': {
        'min': int(tok_len.min()),
        'median': int(np.median(tok_len)),
        'mean': round(float(tok_len.mean()), 1),
        'p75': int(np.percentile(tok_len, 75)),
        'p90': int(np.percentile(tok_len, 90)),
        'p95': int(np.percentile(tok_len, 95)),
        'p99': int(np.percentile(tok_len, 99)),
        'max': int(tok_len.max()),
        'fertility': round(float(tok_len.mean() / char_len.mean()), 4),
        'n_leq_256': int((tok_len <= 256).sum()),
        'pct_leq_256': round(float((tok_len <= 256).mean() * 100), 1),
        'n_leq_512': int((tok_len <= 512).sum()),
        'pct_leq_512': round(float((tok_len <= 512).mean() * 100), 1),
        'n_gt_512': int((tok_len > 512).sum()),
        'pct_gt_512': round(float((tok_len > 512).mean() * 100), 1),
    },
    'structural_features': {
        'n_atoms': {
            'mean': round(float(df['n_atoms'].mean()), 1),
            'median': int(df['n_atoms'].median()),
            'max': int(df['n_atoms'].max()),
        },
        'n_bonds': {
            'mean': round(float(df['n_bonds'].mean()), 1),
            'median': int(df['n_bonds'].median()),
            'max': int(df['n_bonds'].max()),
        },
        'n_reactant_templates': df['n_reactant_templates'].value_counts().sort_index().to_dict(),
        'n_product_templates': {int(k): int(v) for k, v in
                                df['n_product_templates'].value_counts().sort_index().items()},
    },
    'vocabulary': {
        'rule_based_size': vocab['size'],
    },
    'ec_annotations': {
        'n_with_ec': int(has_ec.sum()),
        'pct_with_ec': round(float(has_ec.mean() * 100), 1),
        'n_without_ec': int((~has_ec).sum()),
        'pct_without_ec': round(float((~has_ec).mean() * 100), 1),
        **ec_stats,
    },
    'train_val_split': {
        'val_fraction': 0.1,
        'n_train': 325576,
        'n_val': 36175,
    },
}

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"\nSaved results to {OUTPUT_JSON}")
