# Running the project

## Setup

Install Python 3.11 or newer. Open the project folder in VS Code, choose **Terminal → New Terminal**, and create a local environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe run_analysis.py
```

This downloads prices, builds the portfolio, estimates all five models at 95% and 99%, performs the backtests, and saves tables and figures. Opening the folder does not start the analysis automatically.

The verified full run took approximately **16 minutes**, excluding the download. Reference summary tables are included in `results/`; generated runs are stored locally under `outputs/`.

For a faster run with two models:

```powershell
.\.venv\Scripts\python.exe run_analysis.py --models historical normal
```

Alternatively, select **Ctrl+Shift+P → Python: Select Interpreter → .venv**, open `run_analysis.py`, and click **Run Python File in Terminal**. Run from the project root. There is no need to execute the individual model files.

## Understanding the virtual environment

`.venv` is a local Python environment for this project. It provides a Python executable and a place to install dependencies such as `arch` and `yfinance`. Calling `.venv\Scripts\python.exe` selects it explicitly, so shell activation is unnecessary.

A standard environment created with `python -m venv .venv` keeps its installed packages separate from other projects.

The environment is not a virtual machine, cloud service, or background process. It does not start the analysis or upload anything. `.venv/` is excluded from Git because each user creates an environment on their own computer using the dependencies declared in `pyproject.toml`.

## Finding the results

Each run creates a separate directory in `outputs/`:

- `backtesting_95%_formatted.csv` and `backtesting_99%_formatted.csv`: readable tables;
- `backtesting_results.csv`: full numeric results;
- `figures/`: PNG charts;
- `forecasts/`: daily forecasts and exceedances;
- `convergence_diagnostics.csv`: dates, model names and optimiser messages for non-converged GARCH fits; header-only when none were reported;
- `manifest.json`: settings and dependency versions, written after successful completion.

Charts are saved instead of displayed in pop-up windows. CSV files can be opened in Excel. Percentages in formatted tables are presentation strings; the full numeric table stores decimal values.

## Understanding the modules

`config.py` holds study settings. `data.py` downloads and validates prices, and `portfolio.py` constructs the portfolio. `models/` contains the five VaR methods. `backtesting.py` evaluates forecasts, `reporting.py` formats tables, and `plotting.py` saves charts. `pipeline.py` connects these steps; `run_analysis.py` is the entry point.

Defaults preserve the original study: 2005–2025, a 252-day window, 95% and 99% confidence levels, and USD 1 million initial capital. Changing these settings defines a different experiment. Revised Yahoo data and different dependency versions can change results relative to the submission.

## Alternative setup command

If `python` is not on the current computer's PATH but Anaconda is installed in its default per-user location, use:

```powershell
& "$env:USERPROFILE\anaconda3\python.exe" -m venv .venv
```

Run the tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Original notebook backups are stored locally in `.local-backup/`, which Git ignores. The dissertation PDF is in `docs/dissertation/`. Running the analysis does not create a Git commit or push changes to GitHub.

Rolling exceedance charts require at least 253 forecast observations and use a fixed 252-day diagnostic window. Short runs skip these two charts with a warning. If GARCH emits a convergence warning, inspect `convergence_diagnostics.csv`; affected forecasts are retained for comparison with the dissertation, not certified as reliable fits.
