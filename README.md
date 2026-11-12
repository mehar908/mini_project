# Banking Fraud Detection

A Flask dashboard that replays labeled transaction data, scores transactions
with several classifiers, compares model performance, and stores processed
history and investigation reviews in SQLite. This is a demonstration project,
not a live payment processor or a production fraud decision system.

## Technology stack

- Python 3.13+
- Flask for the dashboard and JSON/report endpoints
- pandas and NumPy for dataset preparation
- scikit-learn for Logistic Regression, Decision Tree, Random Forest, and
  Isolation Forest
- XGBoost comparison when `xgboost` is installed
- SQLite for transaction history and investigation records
- Matplotlib for PDF report generation
- HTML, CSS, and JavaScript for the dashboard

## Folder structure

```text
Banking Fraud Detection/
├── app.py                 # Flask app, training, scoring, evaluation, reports
├── creditcard.csv         # Required labeled input dataset (Class target)
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment configuration
├── .gitattributes         # Git LFS tracking for creditcard.csv
├── static/style.css       # Dashboard styles
├── templates/
│   ├── index.html         # Overview and transaction summary
│   ├── live_stream.html   # Filterable scored replay
│   ├── rules.html         # Editable demo decision thresholds
│   ├── sandbox.html       # Decision policy simulator
│   ├── disputes_sla.html  # Illustrative labeled-fraud review queue
│   ├── syndicate_hunter.html # Repeated model-signal explorer
│   └── methodology.html   # Methodology documentation page
├── transactions.sqlite3   # Created on first run; ignored by Git
└── 01–04_*.py             # Standalone exploration/scoring examples
```

## Requirements and installation

Install Python 3.13 or newer. From PowerShell in this project folder:

```powershell
py -3 -m pip install -r requirements.txt
```

The `creditcard.csv` dataset must be beside `app.py` and contain a numeric
binary `Class` column. XGBoost is optional; when it is not installed, the
comparison table and reports state that it was unavailable.

## Run the application

```powershell
py -3 app.py
```

Open <http://127.0.0.1:5000>. Stop the server with `Ctrl+C`. The application
binds to `0.0.0.0` and uses the `PORT` environment variable when provided by a
host.

## Train and evaluate models

Model training and holdout evaluation run during application startup. The
dashboard uses the same stratified 80/20 holdout set to compare Logistic
Regression, Decision Tree, Random Forest, Isolation Forest, the Random
Forest/Isolation Forest hybrid, and XGBoost when available. `/api/summary`
returns the calculated model comparison.

The data is sampled reproducibly to at most 20,000 rows (`random_state=42`).
Feature missing values are imputed using medians from the training partition.
Supervised models use class balancing and generated fraud examples based only
on training fraud rows. Isolation Forest is fit on non-fraud training rows.
The hybrid score is `0.60 × Random Forest fraud probability + 0.40 ×
Isolation Forest anomaly percentile`. Evaluation uses a 0.50 threshold for
class predictions and reports accuracy, precision, recall, F1-score,
ROC-AUC, average precision (PR-AUC), confusion matrix, false-positive rate,
and false-negative rate.

## Reports and methodology

Download generated results from the dashboard or directly:

- CSV: `/reports/csv`
- PDF: `/reports/pdf`
- Methodology page: `/methodology`

Reports are generated from the loaded dataset, current model evaluations,
measured training/inference timings, environment details, and current database
statistics. They include dataset statistics, model descriptions, methodology,
comparison metrics and confusion matrices, system performance, hybrid results,
limitations, and suggested future improvements.

## Database

SQLite is stored at `transactions.sqlite3` beside `app.py` by default. Override
the location with `DATABASE_PATH`. The `transactions` table stores a stable
transaction ID, amount, timestamp, original label when available, model scores,
predicted class, risk category, and investigation status. The `investigations`
table stores review events, notes, and review timestamps. Stable IDs prevent
duplicate records; each replay updates a separate last-replayed timestamp so
the live history continues moving without changing the original transaction
timestamp or duplicating dataset rows. Existing JSON transaction history is
migrated when the app starts.

The dashboard supports transaction history search and risk-category filters,
investigation status/notes updates, and displays actual database record count,
fraud alerts, reviewed transactions, and SQLite file size.

## Workspace pages and simulated decisions

The top navigation separates the Overview, Live Stream, Rules, Sandbox,
Disputes SLA, and AI SyndicateHunter pages. Replay decisions use editable
thresholds stored in SQLite: scores below the approval threshold are marked
Approved, scores from there to the block threshold are marked Challenged 3DS,
and scores at or above the block threshold are marked Blocked. Changing the
rules recalculates decisions for transactions already stored in the replay
database. In Review is a separate manual investigation status.

These are demonstrations, not processor outcomes. The provided CSV has
transaction features and fraud labels, but no real authorization, 3-D Secure,
chargeback, customer, card, or merchant events. The SLA page uses an
illustrative four-hour review timer for replayed dataset-labeled fraud. The
SyndicateHunter page groups repeated model-explanation features and does not
identify or link people or merchants.

The supplementary rules page includes Tor/VPN exit blocking, card-testing
velocity, and impossible-travel checks. `Amount`, `Time`, and `Class` support
only global low-amount velocity and amount-versus-average proxies. The CSV
does not include IP/Tor/VPN reputation, card/device/session identifiers,
country, coordinates, or account age. Rule candidates do not change the
dataset's `Class` label or automatically alter the simulated payment decision.
No new columns are added to the source CSV; unavailable signals must come
from a separately supplied, privacy-reviewed transaction source.

The Live Stream prepends six labeled synthetic rule scenarios: individual Tor,
card-testing, and impossible-travel matches, plus clean/non-matching controls.
Their model features are based on legitimate rows from the dataset, while
external signal values are fictional test inputs. Their dataset fraud label
remains `Synthetic scenario`; the original CSV `Class` values are preserved
for actual dataset transactions. The stream requests another replay batch
every 2.5 seconds while running.

## Limitations

- Dataset provenance and representativeness are not independently verified.
- The model comparison uses a reproducible sample of at most 20,000 rows.
- Generated fraud examples and the anomaly percentile are demonstration
  techniques; risk scores are not calibrated financial probabilities.
- System timings vary by machine, installed package versions, and load.
- Public deployment requires appropriate access controls and persistent
  storage. SQLite files on ephemeral hosting storage may be lost at restart.
