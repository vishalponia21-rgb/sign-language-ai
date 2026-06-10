import csv
import random
import numpy as np
import os

INPUT_CSV = 'dataset/hand_data.csv'
SAMPLES_PER_LABEL = 100
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

if not os.path.exists(INPUT_CSV):
    raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

with open(INPUT_CSV, 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)

header = rows[0]
data = rows[1:]

labels = {}
for row in data:
    if not row:
        continue
    lbl = row[0]
    labels.setdefault(lbl, []).append(row)

all_labels = [chr(i) for i in range(65, 91)]

print('Current counts:')
for L in all_labels:
    print(f"  {L}: {len(labels.get(L, []))}")

new_rows = []
# Helper to jitter numeric values
def jitter_values(values, scale=0.02):
    out = []
    for v in values:
        try:
            fv = float(v)
        except:
            fv = 0.0
        nv = fv + np.random.normal(0, scale)
        # clamp x,y to [0,1]
        out.append(round(float(max(min(nv, 1.0), -1.0)), 4))
    return out

# Flatten all existing numeric rows to use as donors if a label has zero samples
all_existing_rows = [r for r in data if r]
if not all_existing_rows:
    raise SystemExit('No existing samples to synthesize from.')

for L in all_labels:
    current = labels.get(L, [])
    need = SAMPLES_PER_LABEL - len(current)
    if need <= 0:
        continue
    print(f"Filling {L}: need {need} samples")
    for i in range(need):
        if len(current) > 0:
            donor = random.choice(current)
        else:
            donor = random.choice(all_existing_rows)
        numeric = donor[1:]
        jittered = jitter_values(numeric, scale=0.02)
        new_row = [L] + [f"{v:.4f}" for v in jittered]
        new_rows.append(new_row)

# Append to CSV
if new_rows:
    with open(INPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        for r in new_rows:
            writer.writerow(r)

# Report final counts
with open(INPUT_CSV, 'r') as f:
    reader = csv.reader(f)
    rows2 = list(reader)[1:]

final_counts = {}
for row in rows2:
    if not row:
        continue
    final_counts[row[0]] = final_counts.get(row[0], 0) + 1

print('\nFinal counts:')
for L in all_labels:
    print(f"  {L}: {final_counts.get(L, 0)}")

print('\nDone: dataset augmented to ensure at least 100 samples per letter.')
