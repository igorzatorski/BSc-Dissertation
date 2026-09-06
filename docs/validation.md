# Refactor validation

Validation was performed locally on 6 September 2026. The original notebook was backed up before extraction and numerical functions were frozen in `tests/fixtures/original_functions.py`.

## Completed checks

- `python -m pytest -q`: **18 tests passed**. All five models at 95% and 99% and all backtesting summary fields matched the original functions exactly on a deterministic 272-price-row fixture (252-return calibration window, 18 forecast observations).
- Each of the ten model/confidence cases checks that changing the realised return at the final forecast date does not change that day's threshold.
- Original portfolio alignment, zero/all-breach likelihood handling, incomplete ETF input and output overwrite protection were tested.
- At the initial refactor, an AST comparison confirmed that **18 original non-plotting functions** retained identical executable structure after excluding added docstrings and type annotations.
- A real-data end-to-end smoke run completed for all ten combinations on prices from December 2024 through December 2025, producing ten forecast CSVs, numeric and formatted tables, ten PNG figures and a completion manifest. Its 17 forecast observations validate execution only; they are not a meaningful statistical sample.
- The full Yahoo Finance input download returned **5,283 rows and ten ETFs**, from 3 January 2005 to 31 December 2025.
- Independent known-value tests verify the Historical VaR quantile, closed-form Normal VaR forecast and Kupiec likelihood-ratio statistic.
- Ruff and Black check the application code, tests and entry point. Local Markdown links are checked for missing targets.
- The public PDF has 38 pages and removes the student number from both front-matter occurrences. The remaining research content is unchanged; the private original is excluded from Git.

## Full-sample verification

The full pipeline completed successfully on the freshly downloaded 2005â€“2025 prices, using the original default settings. It produced **5,281 portfolio observations, 5,029 forecasts per combination, ten forecast files and ten PNG figures**, together with all summary tables and the completion manifest. All ten images were checked for file integrity; representative full-sample charts were visually inspected.

The run took **936.93 seconds (15 minutes 37 seconds)**, excluding the separate download. Local results are in `outputs/full-validation/` (excluded from Git). The saved aligned-price file has SHA-256 `18177118155c8d8a5c6c3f6a80864668f5e82b26d7dcbae32c0d72fa6a405d2d`.

| Model | 95% breaches: run / submission | 99% breaches: run / submission |
|---|---:|---:|
| Historical | 293 / 293 | 86 / 86 |
| Normal | 297 / 297 | 136 / 136 |
| Student-t | 346 / 346 | 94 / 94 |
| GARCH-Normal | 337 / 337 | 120 / 120 |
| GARCH-Student-t | 350 / 350 | 96 / 96 |

Kupiec and conditional coverage rejected all ten combinations. Independence rejected Historical, Normal and Student-t at both confidence levels and did not reject either GARCH model. These decisions also agree with the submitted study. Matching these results does not imply that the newly downloaded prices or every floating-point statistic are identical to the original run.

## Environment

The full-sample research run used Python 3.13 on Windows; its manifest records all direct package versions. The public setup instructions were independently tested in a clean Python 3.11 environment with no access to Anaconda site packages. Installation from `pyproject.toml`, all 18 tests, Ruff, Black and a ten-combination end-to-end smoke run completed successfully. macOS and Linux were not tested locally; GitHub Actions provides an independent Ubuntu run after publication.

## Reproduction boundaries

Regression parity establishes that the refactor preserves the supplied notebook's calculations on identical inputs and versions. It does not independently validate every modelling assumption or establish byte-for-byte reproduction of the submitted results on newly downloaded data. GARCH fitting parameters, return alignment and statistical formulas are preserved. Subsequent hardening adds convergence diagnostics without altering forecasts.

Local data and generated validation outputs are ignored by Git. Reference backtesting tables are included in `results/`. The `figures/` directory contains exports from the validated full run, while the privacy-redacted PDF retains the submitted study; the full original price snapshot was not provided.

## Publication hardening

Short runs now skip rolling charts when fewer than 253 forecasts are available. Earlier 17-observation smoke runs produced empty rolling charts; their other eight charts and numerical outputs remain useful execution checks. Each new run exports `convergence_diagnostics.csv` (header-only when no GARCH fit failed to converge), and its manifest records `garch_nonconverged_fits`. Older full-run outputs predate these diagnostics and do not establish convergence of every fit.

The hardening checks passed: **26 tests**, Ruff and Black. A fresh offline smoke run using the saved December 2024–December 2025 prices completed all ten model/confidence combinations with 17 forecasts each. It produced eight non-rolling charts, skipped both rolling charts with explicit warnings, and recorded zero non-converged fits. The full-sample estimation was not repeated for this change; exact numerical regression against the original functions still passes.
