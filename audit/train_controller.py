"""
train_controller.py -- trains the learned CANDOR-FD controller.

MODEL
-----
A pretrained text encoder (DeBERTa-v3-base by default) reads the words
heard so far. Its output is joined with three timing numbers that a
live system would truly know at that moment. A small head then gives
one answer: keep talking, or stop.

The encoder is the right size for this job. The input is only a few
words plus three numbers. A much larger model would not read those few
words any better; it would only take longer and fit the training data
too closely.

CLASS IMBALANCE
---------------
Real turns outnumber backchannels by roughly six to one. Loss weights
correct for this, so the model does not simply learn to answer "stop"
every time and appear accurate while being useless.

WHAT IS SAVED
-------------
The best checkpoint by validation PR-AUC, plus the exact feature
normalisation values used. Those values must travel with the model --
without them the same model would read its own inputs wrongly at
inference time.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

NUMERIC_COLS = ["prefix_word_count", "elapsed_in_speech", "time_into_other_turn", "speech_rate"]
MAX_TOKEN_LEN = 24     # prefixes are at most 5 words; this leaves generous room
WARMUP_STEPS = 200      # fixed length -- independent of how many epochs actually run
MAX_CONSECUTIVE_NONFINITE = 20  # abort the epoch early rather than waste it if this many losses in a row are bad


class PrefixDataset(Dataset):
    def __init__(self, df, tokenizer, feat_mean, feat_std):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].to_numpy().astype(np.float32)
        feats = df[NUMERIC_COLS].to_numpy().astype(np.float32)
        self.feats = (feats - feat_mean) / feat_std
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=MAX_TOKEN_LEN,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "features": torch.tensor(self.feats[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class ControllerModel(nn.Module):
    def __init__(self, encoder_name, n_numeric, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, use_safetensors=True)
        hidden = self.encoder.config.hidden_size
        self.numeric_proj = nn.Sequential(
            nn.Linear(n_numeric, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden + 64, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask, features):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DeBERTa has no pooler layer; the first token's vector is the
        # standard sentence representation to use here.
        cls = out.last_hidden_state[:, 0]
        joined = torch.cat([cls, self.numeric_proj(features)], dim=1)
        return self.head(joined).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device, use_bf16):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["features"].to(device),
            )
        all_logits.append(logits.float().cpu())
        all_labels.append(batch["label"])
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    if not np.isfinite(logits).all():
        n_bad = int((~np.isfinite(logits)).sum())
        print(f"  WARNING: {n_bad:,} non-finite predictions during evaluation -- "
              f"the model has diverged. Reporting this epoch as failed rather than crashing.")
        return {"pr_auc": 0.0, "roc_auc": 0.0, "diverged": True}

    probs = 1 / (1 + np.exp(-logits))
    return {
        "pr_auc": float(average_precision_score(labels, probs)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "diverged": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--outdir", default="checkpoints")
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base",
                    help="swap to roberta-base if the tokenizer errors out")
    ap.add_argument("--max-epochs", type=int, default=15,
                    help="upper bound only -- training stops earlier via --patience "
                         "once validation stops improving")
    ap.add_argument("--patience", type=int, default=3,
                    help="stop after this many epochs with no validation improvement")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # DeBERTa-v3's disentangled attention has a documented numerical
    # instability under bf16: gradient magnitudes underflow in bf16's
    # short 7-bit mantissa. This matches the observed failure exactly --
    # NaN from the very first step, before any real training happened,
    # independent of learning rate. Other encoders (e.g. roberta-base)
    # use standard attention and don't share this issue, so bf16 stays
    # available for those.
    use_bf16 = (
        device == "cuda"
        and torch.cuda.is_bf16_supported()
        and "deberta" not in args.encoder.lower()
    )
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  bfloat16: {use_bf16}")
    else:
        print("WARNING: no GPU found -- this will be very slow.")

    datadir = Path(args.datadir)
    train_df = pd.read_parquet(datadir / "train.parquet")
    val_df = pd.read_parquet(datadir / "val.parquet")
    print(f"train: {len(train_df):,}   val: {len(val_df):,}")

    # Normalisation comes from TRAINING data only. Using the whole set
    # would leak information from validation into training.
    feat_mean = train_df[NUMERIC_COLS].to_numpy().astype(np.float32).mean(axis=0)
    feat_std = train_df[NUMERIC_COLS].to_numpy().astype(np.float32).std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    train_ds = PrefixDataset(train_df, tokenizer, feat_mean, feat_std)
    val_ds = PrefixDataset(val_df, tokenizer, feat_mean, feat_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = ControllerModel(args.encoder, len(NUMERIC_COLS)).to(device).float()
    print(f"Model parameter dtype (must be torch.float32): {next(model.parameters()).dtype}")

    n_pos = int((train_df["label"] == 1).sum())
    n_neg = int((train_df["label"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"STOP: {n_pos:,}   BACKCHANNEL: {n_neg:,}   pos_weight: {pos_weight.item():.3f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    encoder_params = list(model.encoder.parameters())
    head_params = list(model.numeric_proj.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=0.01)
    base_lrs = [args.lr, args.head_lr]

    def apply_warmup_lr(step):
        """Fixed-length linear warmup -- reaches peak LR at WARMUP_STEPS and
        HOLDS there, regardless of how many total steps training ends up
        running for. (The earlier version scaled warmup length to
        --max-epochs, which meant the LR was still climbing, not yet
        annealing, at the exact point training diverged.)"""
        scale = min((step + 1) / WARMUP_STEPS, 1.0)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * scale

    # After warmup, this reduces LR when validation PR-AUC stops improving --
    # tied to the SAME signal driving early stopping, instead of guessing a
    # decay shape upfront for a run length that isn't known in advance.
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-7,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_pr_auc = -1.0
    epochs_since_improvement = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running = 0.0
        consecutive_nonfinite = 0
        diverged = False
        for step, batch in enumerate(train_loader, 1):
            global_step = (epoch - 1) * len(train_loader) + step
            apply_warmup_lr(global_step)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                    batch["features"].to(device, non_blocking=True),
                )
                loss = loss_fn(logits.float(), batch["label"].to(device, non_blocking=True))

            if not torch.isfinite(loss):
                consecutive_nonfinite += 1
                print(f"  WARNING: non-finite loss at epoch {epoch} step {step} -- "
                      f"skipping this batch, not corrupting the model with it")
                if consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE:
                    print(f"\n  {consecutive_nonfinite} non-finite losses in a row -- "
                          f"the model has diverged. Stopping this epoch now instead of "
                          f"wasting the rest of it on a broken model.")
                    diverged = True
                    break
                continue
            consecutive_nonfinite = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)}  loss {running/200:.4f}  "
                      f"lr(head)={optimizer.param_groups[1]['lr']:.2e}")
                running = 0.0

        if diverged:
            metrics = {"pr_auc": 0.0, "roc_auc": 0.0, "diverged": True}
        else:
            metrics = evaluate(model, val_loader, device, use_bf16)
        print(f"epoch {epoch}: val PR-AUC {metrics['pr_auc']:.4f}  ROC-AUC {metrics['roc_auc']:.4f}")

        if global_step >= WARMUP_STEPS and not metrics.get("diverged"):
            plateau_scheduler.step(metrics["pr_auc"])

        if metrics["pr_auc"] > best_pr_auc and not metrics.get("diverged"):
            best_pr_auc = metrics["pr_auc"]
            epochs_since_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "encoder_name": args.encoder,
                "numeric_cols": NUMERIC_COLS,
                "feat_mean": feat_mean.tolist(),
                "feat_std": feat_std.tolist(),
                "val_metrics": metrics,
                "epoch": epoch,
            }, outdir / "best_model.pt")
            (outdir / "training_summary.json").write_text(json.dumps({
                "best_epoch": epoch, "best_val_pr_auc": best_pr_auc,
                "val_metrics": metrics, "encoder": args.encoder,
            }, indent=2))
            print(f"  saved new best -> {outdir/'best_model.pt'}")
        else:
            epochs_since_improvement += 1
            print(f"  no improvement ({epochs_since_improvement}/{args.patience})")
            if epochs_since_improvement >= args.patience:
                print(f"\nStopping early: no validation improvement for "
                      f"{args.patience} epochs in a row.")
                break

    print(f"\nDone. Best validation PR-AUC: {best_pr_auc:.4f}")
    print("The saved checkpoint is from the BEST epoch, not necessarily the last one run.")
    print("Next: run evaluate.py to compare this against the rule-based controller.")


if __name__ == "__main__":
    main()