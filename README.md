# Institutional Crowding and Equity Fragility

This repository contains the code and report for a research project examining whether institutional ownership and network structure improve the prediction of stock-level downside risk beyond market information.

**Pipeline:** [`scripts/run_all.py`](scripts/run_all.py)  
**Report:** [`report/report.pdf`](report/report.pdf)

## Overview

SEC Form 13F holdings are combined with CRSP market data to construct a quarterly stock-level panel covering 2013-2025. The primary task predicts each stock's maximum drawdown over the following 63 trading days.

Ridge and XGBoost are compared across market, ownership, and engineered network feature sets. GraphSAGE, GAT, and Temporal GraphSAGE are also evaluated directly on the quarterly manager-stock ownership graph. Model development uses chronological training and validation samples, followed by one held-out test evaluation. Predictions from the selected model are converted into a within-quarter Fragility Score.

## Key Findings

- A market-only XGBoost model provides the preferred balance between predictive performance and model complexity.
- Conventional ownership measures provide only small incremental improvements, while the engineered network features do not produce a consistent additional gain.
- Temporal GraphSAGE is competitive but does not consistently outperform the simpler tabular model across evaluation metrics.
- The Fragility Score strongly ranks subsequent downside risk, while its portfolio relevance is more sensitive to weighting, the short test period, and implementation assumptions.

## Repository Structure

```text
config/       Project paths, data sources, and fixed settings
data/         Data manifests and local data directories
models/       Saved model artifacts
notebooks/    Data audits, diagnostics, and interpretation
report/       Compiled project report
results/      Generated figures and tables
scripts/      Numbered pipeline entry points
src/          Reusable project code
tests/        Unit tests for core transformations
```

## Requirements and Installation

The project requires Python 3.12 or newer.

```bash
git clone https://github.com/ansabu01/institutional-crowding-equity-fragility.git
cd institutional-crowding-equity-fragility
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a local `.env` file from `.env.example` and provide the required credentials:

```dotenv
SEC_CONTACT_EMAIL=
WRDS_USERNAME=
WRDS_PASSWORD=
```

## Data Access

The project uses SEC Form 13F structured holdings data, CRSP security and return data accessed through WRDS, and daily factors from the Kenneth French Data Library. CRSP and WRDS data require authorized access and must be obtained independently. Their licensing terms prohibit redistributing the underlying restricted data.

## Usage

Run the project smoke test first:

```bash
python scripts/00_smoke_test.py
```

Run the complete numbered pipeline:

```bash
python scripts/run_all.py
```

The pipeline contains 31 entry points numbered from `00` to `30`. Individual scripts can also be run separately in numerical order.

## Reproducibility

The project uses fixed source manifests and SHA-256 checks, deterministic random seeds, point-in-time information dates, and chronological training, validation, and test samples. Raw data are treated as immutable. Full reproduction requires authorized CRSP access and the environment variables listed above.

## License

The software code is released under the [MIT License](LICENSE). The licence does not cover the report text, restricted datasets, institutional branding, or third-party materials.
