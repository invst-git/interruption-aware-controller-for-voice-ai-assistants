"""
check_features.py -- checks the ALREADY-BUILT train/val/test parquet
files for NaN, Inf, or extreme outlier values in the numeric features.
Fast, no GPU needed, reads files that already exist on disk.

Run this before touching the training script again -- it tells us
definitively whether a bad feature value caused the NaN loss, instead
of guessing.
"""
import numpy as np
import pandas as pd

NUMERIC_COLS = ["prefix_word_count", "elapsed_in_speech", "time_into_other_turn", "speech_rate"]

for split in ["train", "val", "test"]:
    df = pd.read_parquet(f"data/{split}.parquet")
    print(f"\n=== {split} ({len(df):,} rows) ===")
    for col in NUMERIC_COLS:
        vals = df[col].to_numpy(dtype=np.float64)
        n_nan = int(np.isnan(vals).sum())
        n_inf = int(np.isinf(vals).sum())
        finite = vals[np.isfinite(vals)]
        p99 = np.percentile(finite, 99) if len(finite) else float("nan")
        flag = " <-- CHECK THIS" if (n_nan or n_inf or (len(finite) and finite.max() > p99 * 20)) else ""
        print(f"  {col:22} min={finite.min():>12.3f}  max={finite.max():>12.3f}  "
              f"mean={finite.mean():>10.3f}  p99={p99:>10.3f}  nan={n_nan}  inf={n_inf}{flag}")
