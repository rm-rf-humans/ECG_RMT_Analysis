# Speaker Notes

## One-minute explanation

This project starts from PTB-XL raw ECG waveform files and metadata. The pipeline maps SCP diagnostic statements into a binary normal-versus-diagnostic-abnormality target, extracts per-lead time-domain features, spectral features, cross-lead correlations, and metadata features, then compares six classifier families on the official PTB-XL folds. XGBoost gives the highest held-out ROC-AUC at 0.9097. The second part of the project uses the Marchenko-Pastur law as a high-dimensional diagnostic. The covariance spectrum has 60 eigenvalues above the MP upper edge, explaining 76.9% of standardized variance. Feeding trained models MP-outlier reconstructions preserves most discrimination, while bulk-only reconstructions are much weaker. Removing the bulk does not improve the best model, so RMT is useful mainly as an analysis tool for this feature map.

## Talking points

- The label is a binary screening abstraction over PTB-XL diagnostic superclasses.
- Preprocessing is fit only on training folds to avoid leakage.
- Metadata missingness is high for height and weight, while decoded waveform features are almost complete.
- Cross-lead correlations have the strongest average univariate association with the target.
- The model comparison covers linear, bagged-tree, randomized-tree, and boosted-tree families.
- RMT is not introduced as another classifier.
- The MP bulk is weak when isolated, but not pure noise.

## Likely questions

**Why not use a deep ECG network?**  
The project focuses on an interpretable applied-ML pipeline from raw signal decoding through engineered features and high-dimensional spectral diagnostics.

**Is the MP bulk useless?**  
No. Bulk-only reconstructions are weak, but hard bulk removal reduces held-out performance. The bulk contains weak complementary information.

**What is the main limitation?**  
The binary abnormal class merges several diagnostic superclasses, so the task is screening rather than subclass diagnosis.
