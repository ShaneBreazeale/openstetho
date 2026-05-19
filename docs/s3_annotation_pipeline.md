# S3 cardiologist annotation pipeline

End-to-end workflow for converting a trained model checkpoint into a
cardiologist-annotated cycle corpus. The protocol details — sampling
strategy, kappa target, IRB notes — live in
[`s3_annotation_protocol.md`](./s3_annotation_protocol.md). This document
covers the *tooling*.

## Step 1 — score every cycle in the corpus

```
uv run python -m openstetho_model.score_corpus \
    --checkpoint runs/s3_circor_v6/best.pt \
    --wavs $CIRCOR_ROOT \
    --pattern '*.wav' \
    --out runs/s3_circor_v6/cycle_scores.csv \
    --backbone s3cnn_v2 \
    --crop-anchor s2
```

Output `cycle_scores.csv` columns:

| column | description |
|---|---|
| `wav` | absolute path to source WAV |
| `cycle_no` | 0-based index of cycle within the recording |
| `s1_idx`, `s2_idx`, `next_s1_idx` | sample indices of the cycle boundaries |
| `score` | model probability of S3 ∈ [0, 1] |
| `segmenter_confidence` | segmenter's per-recording confidence ∈ [0, 1] |

## Step 2 — stratified subset selection

```
uv run python -m openstetho_model.select_for_annotation \
    --scores runs/s3_circor_v6/cycle_scores.csv \
    --murmur-csv $CIRCOR_ROOT/training_data.csv \
    --n-per-stratum 200 \
    --out runs/s3_circor_v6/annotation_subset.csv
```

Three strata are populated, each balanced across patient-level murmur
status (`Present` / `Absent` / `Unknown` per CirCor):

* `high_positive` — top 10 % by score (model thinks "clearly S3")
* `uncertain` — `0.4 ≤ score ≤ 0.6` (model is on the fence, high info value)
* `high_negative` — bottom 10 % (model thinks "clearly not S3")

Total ≈ 600 cycles. Adjust `--n-per-stratum` to taste.

## Step 3 — export clips for review

```
uv run python -m openstetho_model.export_cycle_clips \
    --selection runs/s3_circor_v6/annotation_subset.csv \
    --out runs/s3_circor_v6/annotation_export
```

Produces a directory the annotation viewer can serve:

```
runs/s3_circor_v6/annotation_export/
    manifest.csv           # 1 row per clip
    clips/
        cycle_000000.wav   # 2 s mono PCG, S2 centered
        cycle_000000.png   # mel-spec image (skipped if matplotlib absent)
        ...
```

`manifest.csv` has the schema the cardiologist will fill in. Each rater
copies the file and appends a `label_s3` column with:

| value | meaning |
|---|---|
| `0` | no S3 |
| `1` | S3 present |
| `9` | unannotatable (segmenter mislocated S1/S2 or signal corrupted) |

## Step 4 — inter-rater agreement

After two cardiologists complete the same subset (independently):

```
uv run python -m openstetho_model.compute_kappa \
    --rater-a labels_alice.csv \
    --rater-b labels_bob.csv \
    --out adjudication_needed.csv
```

Prints Cohen's kappa on binary labels (skipping `9` rows). Writes a CSV
of slugs where the raters disagreed for a senior reviewer to adjudicate.

Gate: if `kappa < 0.7`, revise the viewer / protocol and re-annotate. See
`docs/s3_annotation_protocol.md` for rationale.

## Step 5 — adjudication and gold labels

Senior reviewer fills in the gold label for each disagreement row. The
final gold table is the union of:

* clips both raters agreed on (gold = the agreed label)
* clips reviewed by the senior reviewer (gold = senior's label)
* discard rows where any rater marked `9`

Save as `data/s3_circor_labels.csv` with the schema from
`s3_annotation_protocol.md`.

## Step 6 — supervised re-training on real labels

Once gold labels are available, the synthetic-only training regime becomes
a *pretrain* and the real labels drive the final fine-tune. We do not
have the dataset class for this yet — write it when the labels land,
mirroring `S3CycleDataset` but reading labels from CSV instead of
generating them via injection.
