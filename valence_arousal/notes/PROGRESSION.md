# Model Development Progression  
**IN5550 – Valence & Arousal Prediction**

---

## Project Structure

```
.
├── datasets
│   ├── test.csv
│   └── train.csv
├── eval
│   ├── assets
│   │   └── subtask1-template.csv
│   ├── constants.py
│   ├── eval_interface.py
│   ├── eval.py
│   ├── format_checker.py
│   └── __pycache__
│       ├── constants.cpython-312.pyc
│       ├── constants.cpython-39.pyc
│       ├── eval.cpython-312.pyc
│       ├── eval.cpython-39.pyc
│       ├── format_checker.cpython-312.pyc
│       └── format_checker.cpython-39.pyc
├── logs
│   ├── transformer_emotion-3419589.err
│   ├── transformer_emotion-3419589.out
│   ├── transformer_emotion-3420649.err
│   ├── transformer_emotion-3420649.out
│   ├── transformer_emotion-3452544.err
│   └── transformer_emotion-3452544.out
├── notes
│   ├── PROGRESSION.md
│   └── run.txt
├── outputs
│   ├── submission_hybrid.csv
│   ├── submission_tfidf.csv
│   ├── submission_tfidf_user_split.csv
│   ├── submission_transformer.csv
│   └── submission_transformer_split.csv
└── scripts
    ├── baseline_tfidf.py
    ├── drafts
    │   ├── baseline
    │   │   ├── draft_01_baseline.py
    │   │   ├── draft_02_baseline.py
    │   │   ├── draft_03_baseline.py
    │   │   ├── draft_04_baseline.py
    │   │   └── draft_05_baseline.py
    │   ├── draft_05_hybrid.py
    │   ├── hybrid
    │   │   └── draft_01_hybrid.py
    │   └── transformer
    │       ├── draft_01_transformer.py
    │       ├── draft_02.transformer.py
    │       └── draft_03_transformer.py
    ├── hybrid_model.py
    ├── run_baseline.slurm
    ├── run_transformer.slurm
    └── train_transformer.py
```

---

# Overview of Model Evolution

| Draft | Model Type | Key Idea | Valence (r_comp) | Arousal (r_comp) | Insight |
|------|------------|---------|------------------|------------------|--------|
| 1 | TF-IDF | Text only | 0.597 | 0.447 | Baseline |
| 2 | + Metadata | Structural features | 0.612 | 0.435 | Helps valence |
| 3 | + User + Split | Personalization | 0.605 | 0.489 | Helps arousal |
| 4 | Transformer | Contextual model | 0.630 | 0.397 | Improves within-person |
| 5 | Hybrid | Combine models | **0.661** | **0.513** | Best overall |

---

## Performance Comparison Across Drafts
*Bold values indicate the best performance per metric across all drafts.*
### Valence

| Draft | r_between | r_within | r_composite | mae_between | mae_within | mae_composite |
|------|----------|----------|------------|------------|------------|--------------|
| 1 (TF-IDF) | 0.714 | 0.448 | 0.597 | 0.435 | 0.833 | 0.681 |
| 2 (+ Metadata) | 0.734 | 0.453 | 0.612 | 0.422 | 0.823 | 0.669 |
| 3 (+ User + Split) | 0.725 | 0.450 | 0.605 | 0.414 | 0.808 | 0.653 |
| 4 (Transformer) | 0.721 | 0.516 | 0.630 | 0.420 | 0.817 | 0.662 |
| 5 (Hybrid) | **0.760**<sup>*</sup> | **0.533**<sup>*</sup> | **0.661**<sup>*</sup> | **0.396**<sup>*</sup> | **0.774**<sup>*</sup> | **0.619**<sup>*</sup> |

---

### Arousal

| Draft | r_between | r_within | r_composite | mae_between | mae_within | mae_composite |
|------|----------|----------|------------|------------|------------|--------------|
| 1 (TF-IDF) | 0.564 | 0.311 | 0.447 | 0.260 | 0.547 | 0.414 |
| 2 (+ Metadata) | 0.542 | 0.313 | 0.435 | 0.261 | 0.549 | 0.415 |
| 3 (+ User + Split) | 0.627 | 0.322 | 0.489 | 0.239 | 0.518 | 0.387 |
| 4 (Transformer) | 0.456 | 0.335 | 0.397 | 0.276 | 0.546 | 0.420 |
| 5 (Hybrid) | **0.630**<sup>*</sup> | **0.374**<sup>*</sup> | **0.513**<sup>*</sup> | **0.242** | **0.514** | **0.386** |

---

# Draft 1 — TF-IDF Baseline
*(`scripts/drafts/draft_01.py`)*

### Model
- TF-IDF + Ridge regression

### Key Idea
- Represent text as word frequency features

### Performance
*(`r_coposite`)*
- Valence: 0.597  
- Arousal: 0.447  

### Insight
- Captures general sentiment
- Fails on temporal and user-specific variation  
- Arousal is harder → requires context

---

# Draft 2 — Metadata Features
*(`scripts/drafts/draft_02.py`)*

### Improvements
- Added:
  - text_length
  - word_count
  - is_words
  - collection_phase

### Performance
*(`r_coposite`)*
- Valence: 0.612 ↑  
- Arousal: 0.435 ↓  

### Insight
- Improves global patterns (r_between)
- Limited effect on temporal dynamics

### Bottleneck
- No user-specific modeling

---

# Draft 3 — User Modeling + Split
*(`scripts/drafts/draft_03.py`)*

### Improvements
- Added:
  - user_valence_mean
  - user_arousal_mean
- Split models by `is_words`
- Added intensity features

### Performance
*(`r_coposite`)*
- Valence: 0.605 ↓  
- Arousal: 0.489 ↑  

### Insight
- Strong improvement in arousal (especially r_between)
- Valence slightly over-engineered

### Bottleneck
- Still lacks deep semantic understanding

---

# Draft 4 — Transformer Model
*(`scripts/drafts/draft_04_transformer.py`)*

### Improvements
- Replaced TF-IDF with transformer embeddings
- Combined with:
  - metadata
  - temporal features
  - user baselines

### Performance
*(`r_coposite`)*
- Valence: 0.645 ↑↑  
- Arousal: 0.443 ↓  

### Key Effects
- Strong improvement in **within-person correlation**
- Slight drop in **between-person correlation**

### Insight
- Transformer captures:
  - context
  - phrasing
  - subtle emotional variation

### Bottleneck
- Weak global user differentiation (r_between)

---

# Draft 5 — Hybrid Model (Final)
*(`scripts/drafts/draft_05_transformer.py`)*
*(`scripts/drafts/draft_05_hybrid.py`)*

### Improvements
- Improved transformer training:
  - more epochs
  - lower learning rate
  - validation split
- Reintroduced `is_words` split
- Combined models:
```
Final prediction = 0.6 * Transformer + 0.4 * TF-IDF
```

---

## Performance

### Valence
- r_between: **0.757**  
- r_within: 0.525  
- r_composite: **0.656** (BEST)

### Arousal
- r_between: **0.645**  
- r_within: **0.407** ↑  
- r_composite: **0.536** (BEST)

---

## Key Improvements

- Restores strong **global signal (r_between)** from TF-IDF  
- Retains **temporal sensitivity (r_within)** from transformer  
- Improves both dimensions simultaneously  
- Arousal sees the largest gain  

---

## Final Insight

The hybrid model successfully combines:

- **TF-IDF** → stable lexical patterns  
- **Transformer** → contextual and temporal understanding  

Result:

> Balanced model with optimal performance across both emotional dimensions

---

## Remaining Bottleneck

- Optimal weighting between TF-IDF and transformer not fully explored  
- Further improvements possible via:
  - weight tuning
  - ensembling strategies

---

# Final Conclusion

The progression demonstrates a clear evolution:

1. From simple lexical models  
2. → structured feature engineering  
3. → user-aware modeling  
4. → contextual transformer representations  
5. → hybrid integration of complementary strengths  

The final hybrid model achieves the best overall performance, effectively addressing both within-person emotional dynamics and between-person differences.
