# Changelog

## v0.2.0

- Move supervised feature selection into the model-training pipeline so each cross-validation fold fits its own selector.
- Make Python and C++ k-mer counting use the same ambiguity-base and FASTA identifier rules.
- Correct minus-strand Arabidopsis strict-TF bin numbering and align BED windows with chromosome-boundary handling.
- Prevent stale models from previous training runs from entering current feature-importance summaries, and select the final model family by mean CV performance across seeds.
- Calculate binary and multiclass AUC from class-specific prediction probabilities when available.
- Calculate feature importance consistently with permutation importance on each run's training partition.
- Add optional group-disjoint holdout and group-aware cross-validation through `--group-column`.
- Improve annotation validation, Selene prediction parsing, cache placement, and atomic resource downloads.
- Add regression tests and a GitHub Actions test workflow.
