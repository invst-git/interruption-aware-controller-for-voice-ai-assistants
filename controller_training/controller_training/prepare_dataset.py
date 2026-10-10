"""
prepare_dataset.py -- turns candor_backbiter_master.parquet into
training examples for the learned controller.

THE CORE IDEA
-------------
The live controller must decide "keep talking" or "stop" AS words
arrive, not after the sentence ends. So every training example is a
PREFIX: the first k words of something a person said, labelled with
what that speech turned out to be.

  label 0 = BACKCHANNEL  (it stayed a backchannel -> keep talking)
  label 1 = STOP         (it became a real turn -> stop talking)

TWO HONEST DESIGN CHOICES, STATED PLAINLY
-----------------------------------------
1. Only OVERLAPPING real turns are used as the STOP class. In the live
   app this decision is only ever made while the assistant is already
   speaking, so a turn that began in clean silence is not a case the
   controller ever sees. Backchannels are overlapping by definition.
   Restricting turns the same way keeps both classes in the same real
   situation, instead of letting the model separate them on context
   that would never exist live.

2. Only backchannels with backchannel_count == 1 are used. When a turn
   holds several backchannels, the text is joined together and the
   start/stop times span the whole group, not one utterance. Those rows
   cannot give a clean single example.

FEATURE HONESTY
---------------
Every numeric feature must be knowable AT THE MOMENT of the decision.
The `overlap` flag, final word count, and total duration are all
excluded, because a live system does not have them yet.

`elapsed_in_speech` is APPROXIMATED: word timing inside one utterance
is estimated by spreading the duration evenly across its words. The
raw per-word timings do exist (in each conversation's
transcribe_output.json) and would be more exact. That upgrade is left
as a later step; this file marks the approximation clearly rather than
hiding it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MAX_PREFIX_WORDS = 5     # decisions past this point are already obvious
RANDOM_SEED = 42

# Real data contains rare, extreme timing values (a gap of 52+ minutes
# between turns; 1-5 words stretched across 139 seconds) that describe
# situations the live app can never actually be in -- replies are short
# by construction (LLM_MAX_TOKENS), so the assistant is never mid-reply
# for that long. These caps keep training matched to the live app's
# real operating range, not a hack to silence NaN loss.
ELAPSED_IN_SPEECH_CAP_S = 15.0   # ~4x the real 99th percentile (~3.5s)
TIME_INTO_OTHER_TURN_CAP_S = 90.0  # ~2x the real 99th percentile (~44-47s)


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _make_prefix_rows(text, label, time_into_other_turn, speech_duration,
                      conversation_id, source_id):
    """Builds one row per prefix length for a single utterance."""
    words = text.split()
    if not words:
        return []

    n_words = len(words)
    duration = speech_duration if (speech_duration and speech_duration > 0) else 0.0

    rows = []
    for k in range(1, min(n_words, MAX_PREFIX_WORDS) + 1):
        # Approximation: assume words are evenly spread across the
        # utterance. Real per-word timings would replace this.
        elapsed = (k / n_words) * duration
        rows.append({
            "conversation_id": conversation_id,
            "source_id": source_id,
            "text": " ".join(words[:k]),
            "prefix_word_count": k,
            "elapsed_in_speech": elapsed,
            "time_into_other_turn": time_into_other_turn,
            "speech_rate": k / max(elapsed, 0.05),
            "label": label,
        })
    return rows


def build_examples(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["conversation_id", "turn_id"]).reset_index(drop=True)
    df["prev_start"] = df.groupby("conversation_id")["start"].shift(1)

    rows = []

    # ---- STOP class: real turns that began while someone else spoke ----
    turns = df[(df["overlap"] == True) & df["prev_start"].notna()]
    for r in turns.itertuples(index=False):
        text = _clean_text(r.utterance)
        if not text:
            continue
        time_into_other = float(r.start) - float(r.prev_start)
        if time_into_other < 0:
            continue  # inconsistent timing -- skip rather than guess
        rows.extend(_make_prefix_rows(
            text=text,
            label=1,
            time_into_other_turn=time_into_other,
            speech_duration=float(r.delta) if pd.notna(r.delta) else 0.0,
            conversation_id=r.conversation_id,
            source_id=f"turn_{r.conversation_id}_{r.turn_id}",
        ))

    # ---- BACKCHANNEL class: single, cleanly-timed backchannels ----
    bcs = df[(df["backchannel_count"] == 1) & df["backchannel"].notna()]
    for r in bcs.itertuples(index=False):
        text = _clean_text(r.backchannel)
        if not text:
            continue
        if pd.isna(r.backchannel_start) or pd.isna(r.backchannel_stop):
            continue
        time_into_other = float(r.backchannel_start) - float(r.start)
        if time_into_other < 0:
            continue
        rows.extend(_make_prefix_rows(
            text=text,
            label=0,
            time_into_other_turn=time_into_other,
            speech_duration=float(r.backchannel_stop) - float(r.backchannel_start),
            conversation_id=r.conversation_id,
            source_id=f"bc_{r.conversation_id}_{r.turn_id}",
        ))

    examples = pd.DataFrame(rows)

    n_elapsed_clipped = int((examples["elapsed_in_speech"] > ELAPSED_IN_SPEECH_CAP_S).sum())
    n_gap_clipped = int((examples["time_into_other_turn"] > TIME_INTO_OTHER_TURN_CAP_S).sum())
    examples["elapsed_in_speech"] = examples["elapsed_in_speech"].clip(0, ELAPSED_IN_SPEECH_CAP_S)
    examples["time_into_other_turn"] = examples["time_into_other_turn"].clip(0, TIME_INTO_OTHER_TURN_CAP_S)
    examples["speech_rate"] = examples["prefix_word_count"] / examples["elapsed_in_speech"].clip(lower=0.05)
    print(f"  clipped elapsed_in_speech: {n_elapsed_clipped:,} rows "
          f"({n_elapsed_clipped/len(examples)*100:.2f}%)")
    print(f"  clipped time_into_other_turn: {n_gap_clipped:,} rows "
          f"({n_gap_clipped/len(examples)*100:.2f}%)")

    return examples


def split_by_conversation(examples: pd.DataFrame, seed=RANDOM_SEED):
    """Splits on CONVERSATION, never on single rows. If one conversation
    appeared in both training and test data, the model could score well
    by half-remembering it, which would not be a real result."""
    conv_ids = np.array(sorted(examples["conversation_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(conv_ids)

    n = len(conv_ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_ids = set(conv_ids[:n_train])
    val_ids = set(conv_ids[n_train:n_train + n_val])
    test_ids = set(conv_ids[n_train + n_val:])

    return (
        examples[examples["conversation_id"].isin(train_ids)].reset_index(drop=True),
        examples[examples["conversation_id"].isin(val_ids)].reset_index(drop=True),
        examples[examples["conversation_id"].isin(test_ids)].reset_index(drop=True),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="candor_backbiter_master.parquet")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    print(f"Reading {args.parquet} ...")
    df = pd.read_parquet(args.parquet)
    print(f"  {len(df):,} turns, {df['conversation_id'].nunique():,} conversations")

    print("Building prefix examples ...")
    examples = build_examples(df)
    print(f"  {len(examples):,} prefix examples")
    print(f"  BACKCHANNEL (0): {(examples['label'] == 0).sum():,}")
    print(f"  STOP        (1): {(examples['label'] == 1).sum():,}")

    train, val, test = split_by_conversation(examples)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = outdir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        share = (part["label"] == 0).mean() if len(part) else 0
        print(f"  {name}: {len(part):>9,} rows -> {path}  ({share:.1%} backchannel)")

    # Guard: the same conversation must never land in two splits.
    assert not (set(train["conversation_id"]) & set(test["conversation_id"])), \
        "conversation leaked between train and test"
    print("\nSplit check passed: no conversation appears in more than one split.")


if __name__ == "__main__":
    main()
