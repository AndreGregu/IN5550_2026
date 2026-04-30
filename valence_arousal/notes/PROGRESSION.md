# Project Structure and Analysis

## Folder Structure

```
.
└── valence_arousal
    ├── datasets
    │   ├── test.csv
    │   └── train.csv
    ├── eval
    │   ├── assets
    │   │   └── subtask1-template.csv
    │   ├── constants.py
    │   ├── eval_interface.py
    │   ├── eval.py
    │   ├── format_checker.py
    │   └── __pycache__
    │       ├── constants.cpython-312.pyc
    │       ├── eval.cpython-312.pyc
    │       └── format_checker.cpython-312.pyc
    ├── logs
    │   └── valence_baseline-3403830.out
    ├── notes
    │   ├── progression.txt
    │   └── run.txt
    ├── outputs
    │   ├── submission_tfidf.csv
    │   └── submission_tfidf_user_split.csv
    ├── README.md
    └── scripts
        ├── baseline_tfidf.py (main document)
        ├── drafts (previous drafts)
        │   ├── draft_01.py
        │   └── draft_02.py
        └── run_baseline.slurm
```

---

## Draft Overview

| Draft | Model          | Valence     | Arousal            | Insight                       |
| ----- | -------------- | ----------- | ------------------ | ----------------------------- |
| 01    | TF-IDF         | Good        | Weak               | Baseline                      |
| 02    | + Metadata     | Better      | No change          | Structure matters for valence |
| 03    | + User + Split | Slight drop | Strong improvement | User dynamics matter          |

---

## Results

### 1. Evaluation 1 (Draft 01)

#### Performance

**Valence**

* r_between: 0.714 (1.87e-15)
* r_within: 0.448 (6.12e-09)
* r_composite: 0.597
* mae_between: 0.435
* mae_composite: 0.681
* mae_within: 0.833

**Arousal**

* r_between: 0.564 (5.64e-09)
* r_within: 0.311 (5.40e-06)
* r_composite: 0.447
* mae_between: 0.260
* mae_composite: 0.414
* mae_within: 0.547

#### Interpretation

* Learns general sentiment from text
* Fails to capture temporal and user-specific variation
* Arousal is harder to predict than valence

  * Valence: directly expressed in words (e.g. "happy", "sad")
  * Arousal: requires context and intensity understanding

---

### 2. Evaluation 2 (Draft 02)

#### Summary

The model extends TF-IDF with structured metadata features, improving valence prediction but not arousal.

#### Improvements

* Added metadata features:

  * text_length
  * word_count
  * is_words
  * collection_phase
* Model pipeline:

  * TF-IDF(text) + metadata → Ridge regression

#### Rationale

* Text length and word count capture structural differences
* is_words distinguishes essays from word lists
* collection_phase adds weak contextual information

#### Performance

**Valence**

* r_between: 0.734 (1.31e-16)
* r_within: 0.453 (7.10e-09)
* r_composite: 0.612
* mae_between: 0.422
* mae_composite: 0.669
* mae_within: 0.823

**Arousal**

* r_between: 0.542 (2.80e-08)
* r_within: 0.313 (1.67e-05)
* r_composite: 0.435
* mae_between: 0.261
* mae_composite: 0.415
* mae_within: 0.549

#### Key Differences

**Valence**

* r_composite: 0.597 → 0.612
* r_between: 0.714 → 0.734
* r_within: 0.448 → 0.453

**Arousal**

* r_between: 0.564 → 0.542
* r_within: 0.311 → 0.313
* r_composite: 0.447 → 0.435

#### Interpretation

* Improved global pattern recognition across users
* Limited improvement in within-user dynamics
* Metadata does not improve arousal prediction
* Arousal depends more on nuanced semantics than structure

#### Bottleneck

* Modeling user-specific dynamics

---

### 3. Evaluation 3 (Draft 03)

#### Summary

Introduces user-level modeling and dataset splitting, significantly improving arousal prediction.

#### Improvements

* User-level features:

  * user_valence_mean
  * user_arousal_mean
* Dataset split by text type (is_words)
* Separate models per text type
* Intensity-based features:

  * exclamation_count
  * question_count
  * uppercase_count
  * uppercase_ratio

#### Performance

**Valence**

* r_between: 0.725 (4.68e-16)
* r_within: 0.450 (4.04e-09)
* r_composite: 0.605
* mae_between: 0.414
* mae_composite: 0.653
* mae_within: 0.808

**Arousal**

* r_between: 0.627 (3.08e-11)
* r_within: 0.322 (7.10e-06)
* r_composite: 0.489
* mae_between: 0.239
* mae_composite: 0.387
* mae_within: 0.518

#### Key Differences

**Valence**

* r_composite: 0.612 → 0.605
* r_within: 0.453 → 0.450
* r_between: 0.734 → 0.725

**Arousal**

* r_composite: 0.435 → 0.489
* r_within: 0.313 → 0.322
* r_between: 0.542 → 0.627

#### Interpretation

* Strong improvement in arousal prediction
* Especially significant gain in between-user correlation
* Small gains in within-user correlation
* Valence performance slightly decreases

#### Analysis

* Arousal benefits from:

  * User-level modeling
  * Specialized feature engineering
* Valence does not benefit from added complexity
* Splitting data reduces training size and increases variance

#### Bottleneck

* Limited ability to model temporal dynamics
* Lack of deeper semantic understanding beyond surface features

---

## Overall Conclusion

* Valence is largely captured by lexical features
* Arousal requires:

  * Contextual understanding
  * User-specific modeling
  * Intensity-aware features
* Future improvements should focus on:

  * Temporal modeling
  * Richer semantic representations
  * Sequence-based approaches
