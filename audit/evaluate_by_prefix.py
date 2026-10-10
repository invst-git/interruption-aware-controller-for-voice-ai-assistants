"""
evaluate_by_prefix.py -- breaks the neural result down by prefix length.

WHY THIS MATTERS
----------------
The aggregate score mixes two very different situations:

  k = n (the model has heard the WHOLE utterance): the Backbiter label
      is a deterministic function of that full text (<=3 words, lexicon
      majority, allowed starting word). Scoring well here is close to
      tautological -- the label IS a function of the input.

  k = 1 (the model has heard ONE word): this is the actual deployment
      case. The live controller must commit before hearing the rest.

A headline number that averages these together tells you almost
nothing about the real capability. This script separates them.

It also reports PR-AUC with BACKCHANNEL as the positive class, which
evaluate.py does not -- the minority direction is the informative one.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from train_controller import NUMERIC_COLS, ControllerModel, PrefixDataset
from transformers import AutoTokenizer
from evaluate import rule_predicts_stop


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = []
    n_batches = len(loader)
    start_time = time.monotonic()
    for i, batch in enumerate(loader, 1):
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["features"].to(device),
        )
        out.append(logits.float().cpu())
        if i % 10 == 0 or i == n_batches:
            elapsed = time.monotonic() - start_time
            rate = i / elapsed
            remaining_s = (n_batches - i) / rate if rate > 0 else float("inf")
            print(f"  batch {i}/{n_batches}  ({elapsed:.0f}s elapsed, "
                  f"~{remaining_s/60:.1f} min remaining at current rate)")
    return 1 / (1 + np.exp(-torch.cat(out).numpy()))


def metrics_for(y_true, probs, pred_stop):
    is_bc, is_stop = y_true == 0, y_true == 1
    return {
        "n": len(y_true),
        "bc_share": float(is_bc.mean()) if len(y_true) else float("nan"),
        "bc_recall": float((pred_stop[is_bc] == 0).mean() * 100) if is_bc.any() else float("nan"),
        "fp_rate": float((pred_stop[is_stop] == 0).mean() * 100) if is_stop.any() else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = pd.read_parquet(Path(args.datadir) / "test.parquet")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(ckpt["encoder_name"])
    model = ControllerModel(ckpt["encoder_name"], len(NUMERIC_COLS)).to(device).float()
    model.load_state_dict(ckpt["model_state"])

    ds = PrefixDataset(test, tokenizer,
                       np.array(ckpt["feat_mean"], np.float32),
                       np.array(ckpt["feat_std"], np.float32))
    probs = predict_probs(model, DataLoader(ds, batch_size=512, num_workers=2), device)

    y = test["label"].to_numpy()
    neural_stop = (probs >= args.threshold).astype(int)
    rule_stop = np.array([int(rule_predicts_stop(t)) for t in test["text"]])

    print("=" * 78)
    print("PERFORMANCE BY PREFIX LENGTH (k = words heard so far)")
    print("=" * 78)
    print(f"{'k':>3} {'n':>9} {'BC%':>6} | {'RULE recall':>12} {'RULE fp':>9} | "
          f"{'NN recall':>10} {'NN fp':>8}")
    print("-" * 78)
    for k in sorted(test["prefix_word_count"].unique()):
        m = test["prefix_word_count"] == k
        r = metrics_for(y[m], probs[m], rule_stop[m])
        n = metrics_for(y[m], probs[m], neural_stop[m])
        print(f"{k:>3} {r['n']:>9,} {r['bc_share']*100:>5.1f}% | "
              f"{r['bc_recall']:>11.2f}% {r['fp_rate']:>8.2f}% | "
              f"{n['bc_recall']:>9.2f}% {n['fp_rate']:>7.2f}%")

    print("\n  k=1 is the deployment-relevant case: decide from ONE word.")
    print("  k>=4 rows contain zero backchannels by construction (Backbiter")
    print("  defines a backchannel as <=3 words), so they are trivially STOP.")

    print("\n" + "=" * 78)
    print("RANKING QUALITY, BOTH DIRECTIONS")
    print("=" * 78)
    stop_prev = float((y == 1).mean())
    bc_prev = float((y == 0).mean())
    print(f"  PR-AUC, STOP as positive       : {average_precision_score(y, probs):.4f}  "
          f"(random floor = {stop_prev:.4f})")
    print(f"  PR-AUC, BACKCHANNEL as positive: {average_precision_score(1 - y, 1 - probs):.4f}  "
          f"(random floor = {bc_prev:.4f})")
    print(f"  ROC-AUC (direction-independent): {roc_auc_score(y, probs):.4f}")

    print("\n" + "=" * 78)
    print("NON-TRIVIAL SUBSET ONLY (k <= 3, where both classes actually occur)")
    print("=" * 78)
    m = test["prefix_word_count"].to_numpy() <= 3
    print(f"  n = {m.sum():,} of {len(test):,} test prefixes")
    print(f"  PR-AUC, STOP as positive       : {average_precision_score(y[m], probs[m]):.4f}  "
          f"(random floor = {(y[m]==1).mean():.4f})")
    print(f"  PR-AUC, BACKCHANNEL as positive: {average_precision_score(1-y[m], 1-probs[m]):.4f}  "
          f"(random floor = {(y[m]==0).mean():.4f})")
    print(f"  ROC-AUC                        : {roc_auc_score(y[m], probs[m]):.4f}")
    rm = metrics_for(y[m], probs[m], rule_stop[m])
    nm = metrics_for(y[m], probs[m], neural_stop[m])
    print(f"  Rule  : BC recall {rm['bc_recall']:.2f}%  false-pos {rm['fp_rate']:.2f}%")
    print(f"  Neural: BC recall {nm['bc_recall']:.2f}%  false-pos {nm['fp_rate']:.2f}%")

    print("\n" + "=" * 78)
    print("UTTERANCE-LEVEL (one vote per utterance, k=1 only -- independent samples)")
    print("=" * 78)
    m1 = test["prefix_word_count"].to_numpy() == 1
    print(f"  n = {m1.sum():,} independent utterances (no prefix double-counting)")
    print(f"  ROC-AUC: {roc_auc_score(y[m1], probs[m1]):.4f}")
    r1 = metrics_for(y[m1], probs[m1], rule_stop[m1])
    n1 = metrics_for(y[m1], probs[m1], neural_stop[m1])
    print(f"  Rule  : BC recall {r1['bc_recall']:.2f}%  false-pos {r1['fp_rate']:.2f}%")
    print(f"  Neural: BC recall {n1['bc_recall']:.2f}%  false-pos {n1['fp_rate']:.2f}%")


if __name__ == "__main__":
    main()