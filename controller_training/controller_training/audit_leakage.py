"""
audit_leakage.py -- adversarial audit of the controller training result.

This script tries to BREAK the result, not confirm it. Every check is
designed so that a passing result is evidence of validity and a failing
result is concrete evidence of a specific problem.

No GPU or torch needed. Runs on the prepared data/*.parquet files.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

NUMERIC_COLS = ["prefix_word_count", "elapsed_in_speech", "time_into_other_turn", "speech_rate"]

train = pd.read_parquet("data/train.parquet")
val = pd.read_parquet("data/val.parquet")
test = pd.read_parquet("data/test.parquet")

print("=" * 72)
print("CHECK 1 -- conversation-level split integrity")
print("=" * 72)
tr_c, va_c, te_c = (set(d["conversation_id"]) for d in (train, val, test))
print(f"  train/test conversation overlap: {len(tr_c & te_c)}")
print(f"  train/val  conversation overlap: {len(tr_c & va_c)}")
print(f"  val/test   conversation overlap: {len(va_c & te_c)}")
print(f"  VERDICT: {'PASS' if not (tr_c & te_c or tr_c & va_c or va_c & te_c) else 'FAIL -- leakage'}")

print("\n" + "=" * 72)
print("CHECK 2 -- source-utterance overlap (prefixes from the same utterance)")
print("=" * 72)
tr_s, te_s = set(train["source_id"]), set(test["source_id"])
print(f"  train/test source_id overlap: {len(tr_s & te_s)}")
print(f"  test prefixes: {len(test):,} from {test['source_id'].nunique():,} distinct utterances")
print(f"  average prefixes per utterance: {len(test)/test['source_id'].nunique():.2f}")
print("  NOTE: prefixes from one utterance are NOT independent samples.")
print(f"  Effective test size is closer to {test['source_id'].nunique():,} than {len(test):,}.")
print(f"  VERDICT: {'PASS (no cross-split utterance reuse)' if not (tr_s & te_s) else 'FAIL'}")

print("\n" + "=" * 72)
print("CHECK 3 -- is the label deterministic from prefix_word_count alone?")
print("=" * 72)
print("  Backbiter DEFINES a backchannel as <=3 words. If so, any prefix with")
print("  k>=4 can only come from a real turn -- a trivially separable subset.")
for k in sorted(test["prefix_word_count"].unique()):
    sub = test[test["prefix_word_count"] == k]
    bc = (sub["label"] == 0).sum()
    stop = (sub["label"] == 1).sum()
    trivial = " <-- 100% STOP, trivially separable" if bc == 0 else ""
    print(f"  k={k}: {len(sub):>8,} prefixes | BACKCHANNEL {bc:>7,} | STOP {stop:>8,}{trivial}")
trivial_rows = test[test["prefix_word_count"] >= 4]
print(f"\n  Trivially-separable share of test set: {len(trivial_rows)/len(test)*100:.1f}%")
print("  Any aggregate metric is inflated by exactly this share.")

print("\n" + "=" * 72)
print("CHECK 4 -- speech_rate: is it future information?")
print("=" * 72)
per_source_std = test.groupby("source_id")["speech_rate"].std().fillna(0)
constant_share = (per_source_std < 1e-6).mean()
print(f"  Share of utterances where speech_rate is CONSTANT across all prefixes: {constant_share:.1%}")
print("  speech_rate = k / ((k/n)*D) = n/D  -- the FINAL word count divided by")
print("  the FINAL duration. A live system cannot know either at prefix time.")
print(f"  VERDICT: {'CONFIRMED FUTURE-INFORMATION LEAK' if constant_share > 0.9 else 'inconclusive'}")
for lab, name in [(0, "BACKCHANNEL"), (1, "STOP")]:
    s = test[test["label"] == lab]["speech_rate"]
    print(f"    {name:12} speech_rate: mean={s.mean():.3f} median={s.median():.3f}")

print("\n" + "=" * 72)
print("CHECK 5 -- time_into_other_turn: class shortcut?")
print("=" * 72)
print("  Defined as: STOP -> this turn's start minus PREVIOUS turn's start")
print("              BACKCHANNEL -> backchannel start minus its HOST turn's start")
print("  Conceptually the same quantity ('how long has the other been talking'),")
print("  but the two are drawn from structurally different situations.")
for lab, name in [(0, "BACKCHANNEL"), (1, "STOP")]:
    s = test[test["label"] == lab]["time_into_other_turn"]
    print(f"    {name:12} mean={s.mean():>7.3f} median={s.median():>7.3f} "
          f"p25={s.quantile(.25):>6.2f} p75={s.quantile(.75):>7.2f}")

print("\n" + "=" * 72)
print("CHECK 6 -- ABLATION: how far do the NUMERIC FEATURES ALONE get?")
print("=" * 72)
print("  If timing features alone score near the full model, the text encoder")
print("  is not doing the work -- the timing shortcut is.")
scaler = StandardScaler().fit(train[NUMERIC_COLS])
Xtr, ytr = scaler.transform(train[NUMERIC_COLS]), train["label"].to_numpy()
Xte, yte = scaler.transform(test[NUMERIC_COLS]), test["label"].to_numpy()

for name, clf in [
    ("logistic regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ("gradient boosting", HistGradientBoostingClassifier(max_iter=200, random_state=0)),
]:
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    print(f"  {name:22} PR-AUC(STOP+)={average_precision_score(yte, p):.4f}  "
          f"ROC-AUC={roc_auc_score(yte, p):.4f}")

print("\n  ABLATION without speech_rate (the leaking feature):")
cols_no_rate = [c for c in NUMERIC_COLS if c != "speech_rate"]
sc2 = StandardScaler().fit(train[cols_no_rate])
clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
clf.fit(sc2.transform(train[cols_no_rate]), ytr)
p = clf.predict_proba(sc2.transform(test[cols_no_rate]))[:, 1]
print(f"  {'gradient boosting':22} PR-AUC(STOP+)={average_precision_score(yte, p):.4f}  "
      f"ROC-AUC={roc_auc_score(yte, p):.4f}")

print("\n" + "=" * 72)
print("CHECK 7 -- what is the RANDOM baseline for this PR-AUC?")
print("=" * 72)
prev = (yte == 1).mean()
print(f"  Positive class in evaluate.py = STOP, prevalence = {prev:.4f}")
print(f"  A RANDOM classifier scores PR-AUC = {prev:.4f}, not 0.5.")
print(f"  So PR-AUC 0.9981 must be read against a {prev:.3f} floor, not against 0.5.")
print(f"  Headroom actually captured: {(0.9981-prev)/(1-prev)*100:.1f}% of what was available.")
print("\n  PR-AUC with BACKCHANNEL as positive (the minority, harder direction)")
print(f"  would have a random floor of {(yte==0).mean():.4f} -- a far more")
print("  informative framing for this task. evaluate.py does not report it.")

print("\n" + "=" * 72)
print("CHECK 9 -- numeric-only ablation, k=1 ONLY (the actual deployment case)")
print("=" * 72)
print("  Check 6 ran on the full test set, including the trivially-separable")
print("  k>=4 rows -- that could be propping up the 'numeric alone is nearly")
print("  as good' finding artificially. This restricts to k=1: the one case")
print("  that matters live, and the hardest case for numeric features alone")
print("  (elapsed_in_speech is tiny, speech_rate is a noisy 1-word estimate).")
train_k1 = train[train["prefix_word_count"] == 1]
test_k1 = test[test["prefix_word_count"] == 1]
print(f"  train k=1 rows: {len(train_k1):,}   test k=1 rows: {len(test_k1):,}")

sc_k1 = StandardScaler().fit(train_k1[NUMERIC_COLS])
clf_k1 = HistGradientBoostingClassifier(max_iter=200, random_state=0)
clf_k1.fit(sc_k1.transform(train_k1[NUMERIC_COLS]), train_k1["label"].to_numpy())
p_k1 = clf_k1.predict_proba(sc_k1.transform(test_k1[NUMERIC_COLS]))[:, 1]
y_k1 = test_k1["label"].to_numpy()
print(f"  Numeric-only, k=1: PR-AUC(STOP+)={average_precision_score(y_k1,p_k1):.4f}  "
      f"ROC-AUC={roc_auc_score(y_k1,p_k1):.4f}  (random floor={y_k1.mean():.4f})")
print("\n  Compare this ROC-AUC directly against evaluate_by_prefix.py's")
print("  'UTTERANCE-LEVEL (k=1 only)' ROC-AUC for the full neural model.")
print("  Large gap (numeric-only much worse)  -> text genuinely earns its keep.")
print("  Small gap (numeric-only close)       -> text isn't doing much live.")
