# Generated reports

Run `uv run python -m iaces.ml.train_model` to create or replace the model artifact, source-labeled holdout predictions, cross-validation summary, and training manifest. By default, it reads every CSV in `data/processed/`; pass `--data <file>` to train on one file or `--data <file1> <file2> ...` to select explicit inputs.

Then run `uv run python -m iaces.ml.evaluate_model` to create:

- `model_comparison.csv`: overall metrics across the saved evaluation predictions.
- `model_source_comparison.csv`: metrics by model and held-out dataset.
- `model_metrics.json`: both overall and source-specific metrics.
- Updated `training_manifest.json`: evaluation timestamp, protocol, and measured metrics.

With multiple input files, the evaluation leaves one dataset out per fold. With a single file, the original random holdout evaluation is retained. Report the protocol with any metric; the two protocols measure different things.
