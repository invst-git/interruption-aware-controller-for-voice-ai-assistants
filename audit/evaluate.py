"""
evaluate.py -- scores the trained controller against the existing rule,
on the same held-out test conversations, using the same measures.

This is the file that produces the progress-update result. A trained
model on its own proves little. A trained model measured beside the
rule it is meant to replace, on data neither one has seen, is a real
finding.

MEASURES (matching how the rule was scored earlier in this project)
-------------------------------------------------------------------
Recall on backchannels : how often true backchannels are correctly
                         left alone (the assistant keeps talking).
False-positive rate    : how often a real turn is wrongly called a
                         backchannel -- the assistant talks over
                         someone who genuinely wanted to speak. This
                         is the costlier mistake of the two.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from train_controller import NUMERIC_COLS, ControllerModel, PrefixDataset
from transformers import AutoTokenizer

# ---- the existing rule-based controller (v3 lexicon), for comparison ----
BACKCHANNEL_WORDS = {
    "yeah", "mhm", "mm", "hmm", "yep", "yup", "huh", "yes",
    "wow", "oh", "okay", "ah", "exactly", "alright", "gotcha",
}
PROHIBITED_STARTS = {
    "i'm", "i", "no", "wait", "stop", "actually", "but", "sorry", "so",
}


def _clean(word):
    return word.lower().strip(".,!?;:")


def rule_predicts_stop(text):
    words = str(text).split()
    if not words:
        return True
    if _clean(words[0]) in PROHIBITED_STARTS:
        return True
    if len(words) > 3:
        return True
    bc = sum(1 for w in words if _clean(w) in BACKCHANNEL_WORDS)
    return not (bc / len(words) > 0.5)


def score(name, y_true, y_pred_stop):
    y_true = np.asarray(y_true)
    y_pred_stop = np.asarray(y_pred_stop)
    is_bc = y_true == 0
    is_stop = y_true == 1
    recall_bc = float((y_pred_stop[is_bc] == 0).mean()) if is_bc.any() else float("nan")
    fp_rate = float((y_pred_stop[is_stop] == 0).mean()) if is_stop.any() else float("nan")
    return {
        "model": name,
        "backchannel_recall_pct": recall_bc * 100,
        "false_positive_rate_pct": fp_rate * 100,
    }


@torch.no_grad()
def predict_probs(model, loader, device, use_bf16):
    model.eval()
    out = []
    for batch in loader:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["features"].to(device),
            )
        out.append(logits.float().cpu())
    logits = torch.cat(out).numpy()
    return 1 / (1 + np.exp(-logits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    test_df = pd.read_parquet(Path(args.datadir) / "test.parquet")
    print(f"Test set: {len(test_df):,} prefixes from "
          f"{test_df['conversation_id'].nunique():,} held-out conversations\n")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(ckpt["encoder_name"])
    feat_mean = np.array(ckpt["feat_mean"], dtype=np.float32)
    feat_std = np.array(ckpt["feat_std"], dtype=np.float32)

    model = ControllerModel(ckpt["encoder_name"], len(NUMERIC_COLS)).to(device)
    model.load_state_dict(ckpt["model_state"])

    ds = PrefixDataset(test_df, tokenizer, feat_mean, feat_std)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    probs = predict_probs(model, loader, device, use_bf16)
    y_true = test_df["label"].to_numpy()

    neural_stop = (probs >= args.threshold).astype(int)
    rule_stop = np.array([int(rule_predicts_stop(t)) for t in test_df["text"]])

    results = [
        score("Rule-based (v3 lexicon)", y_true, rule_stop),
        score(f"Neural ({ckpt['encoder_name']})", y_true, neural_stop),
    ]

    print(f"{'model':<38} {'BC recall':>12} {'false-pos':>12}")
    print("-" * 64)
    for r in results:
        print(f"{r['model']:<38} {r['backchannel_recall_pct']:>11.2f}% "
              f"{r['false_positive_rate_pct']:>11.2f}%")

    print(f"\nNeural ranking quality (threshold-free):")
    print(f"  PR-AUC : {average_precision_score(y_true, probs):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y_true, probs):.4f}")

    # A single threshold is one operating point, not the whole story.
    # This sweep shows the real trade available from the same model.
    print(f"\nThreshold sweep (choose the point that fits the product):")
    print(f"{'threshold':>10} {'BC recall':>12} {'false-pos':>12}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        s = score("", y_true, (probs >= t).astype(int))
        print(f"{t:>10.2f} {s['backchannel_recall_pct']:>11.2f}% "
              f"{s['false_positive_rate_pct']:>11.2f}%")

    Path("results").mkdir(exist_ok=True)
    Path("results/comparison.json").write_text(json.dumps({
        "results": results,
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "n_test_prefixes": int(len(test_df)),
        "n_test_conversations": int(test_df["conversation_id"].nunique()),
    }, indent=2))
    print("\nSaved results/comparison.json")


if __name__ == "__main__":
    main()
