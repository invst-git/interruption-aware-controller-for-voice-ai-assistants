# Running the Controller Training on qBraid

## GPU to pick

**A single A100.** Not H100, not B200.

The reason is not caution. The input to this model is a few words plus
three numbers. A larger card would not make the model read those words
any better. It would cost more and finish at nearly the same time,
because this job is small in the way that matters. Choosing the right
size for the real job is the correct engineering call, and it is
defensible if anyone asks.

## Step 1 -- get the data file onto your Windows PC

The training only needs ONE file from Google Drive:
`candor_backbiter_master.parquet` (a few tens of MB, already built).

Open Google Drive in your browser, find it inside `CANDOR_processed`,
and download it. You do NOT need to move the 350GB of audio and video.
The training uses none of it.

## Step 2 -- upload files to qBraid

Open qBraid Lab (the same one used for the Pashto project). Make a new
folder, for example `candor-controller`. Upload these five files into
it, using the upload button in the file browser on the left:

- `candor_backbiter_master.parquet`  (from Step 1)
- `prepare_dataset.py`
- `train_controller.py`
- `evaluate.py`
- `check_features.py`  (diagnostic -- run if training ever misbehaves again)
- `requirements.txt`

## Step 3 -- install what is needed

Open a Terminal in qBraid Lab. Then:

```bash
cd candor-controller
pip install -r requirements.txt
```

Then confirm the GPU is truly there and visible, before anything else:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

This must print `True` and an A100 name. If it prints `False`, the
notebook is not on a GPU machine yet. Fix that before going on, or
training will run on CPU and take many hours.

## Step 4 -- build the training data

```bash
python prepare_dataset.py --parquet candor_backbiter_master.parquet --outdir data
```

Takes a few minutes. It prints how many examples it built and how they
split. Check two things in that output:

- Both classes have real counts. Neither should be near zero.
- The last line says the split check passed.

## Step 5 -- train

```bash
python train_controller.py --datadir data --outdir checkpoints
```

No `--epochs` flag anymore -- it trains up to 15 epochs but stops on
its own once validation stops improving for 3 epochs in a row, and
always keeps the best-scoring checkpoint, not just the last one.

It prints the GPU name, whether bfloat16 is on (should say `False` for
the default DeBERTa-v3 encoder -- this is intentional, not a bug; see
the note below), the class balance, and a loss figure every 200 steps.
After each epoch it prints a validation score.

**What a healthy run looks like:** real loss numbers from step 1
onward, no `WARNING: non-finite loss` lines at all. If you see even
one, stop and check `check_features.py` before assuming it will
resolve itself.

If the tokenizer fails to load (a known friction point with this
encoder), switch encoders and run again:

```bash
python train_controller.py --datadir data --outdir checkpoints --encoder roberta-base
```

Both are strong for this task. The second avoids the tokenizer issue,
and also trains in bfloat16 (faster) since it doesn't share DeBERTa-v3's
specific precision sensitivity.

## Step 6 -- the result that matters

```bash
python evaluate.py --datadir data --checkpoint checkpoints/best_model.pt
```

This prints the trained model beside the existing rule, on
conversations neither has seen, scored the same way. It also prints a
threshold sweep, showing the real trade between the two kinds of
mistake.

**This table is your progress update.** It is not "I trained a model."
It is "here is the trained model, here is the rule it must beat, and
here is the measured difference on unseen data."

## Step 7 -- bring the results home

Download `checkpoints/best_model.pt` and `results/comparison.json`
through the qBraid file browser. The checkpoint is what later gets
loaded into the live app, in place of the rule.

## Two honest notes to keep with the results

1. **Word timing inside an utterance is approximated.** Each word is
   assumed to fall evenly across the utterance. The true per-word
   times do exist, inside each conversation's `transcribe_output.json`,
   and would be more exact. That is a real next improvement, not a
   flaw to hide. State it plainly if you present this.

2. **A lower false-positive rate matters more than higher recall
   here.** Talking over someone who genuinely wants to speak is worse
   than pausing for a "yeah" that did not need it. Use the threshold
   sweep in Step 6 to pick that balance on purpose, rather than
   accepting the default 0.5 without thinking about it.

3. **DeBERTa-v3 trains in full fp32, not bfloat16, on purpose.** Its
   attention design has a documented numerical instability under
   bf16 -- gradient values underflow in bf16's short mantissa. This
   costs some speed but is the correct, defensible choice; a
   published benchmark doing the same kind of fine-tuning made the
   same call for the same reason.
