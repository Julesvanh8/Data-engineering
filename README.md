# Data Engineering Project

## Research Question
**How long after a U.S. stock market downturn do unemployment and federal income tax revenues change?**

This project uses an event-study approach to analyze the lagged effects of stock market downturns on macroeconomic indicators.

---

## 🚀 **Quic
- Jules van Halder
- Valerie
- Mohamed

---

## 🎓 **For Instructors**

### **Quick Evaluation:**
1. **View Live Dashboard (No Setup):**  
   https://data-engineering-6urgspnpbupgepmsrsturi.streamlit.app/

2. **Read Comprehensive User Guide:**  
   [USER_GUIDE.md](USER_GUIDE.md) - Complete installation and usage instructions

3. **Run Locally (Optional):**
   ```bash
   git clone https://github.com/Julesvanh8/Data-engineering.git
   cd Data-engineering
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python src/orchestration/run_pipeline.py
   ```

### **Key Deliverables:**
- ✅ Interactive Dashboard (Streamlit Cloud)
- ✅ Complete Pipeline Code (GitHub)
- ✅ DBT Models (SQL transformations)
- ✅ Analysis Results (CSV + visualizations)
- ✅ Comprehensive Documentation

---

## 🔗 Links

- **Live Dashboard:** https://data-engineering-6urgspnpbupgepmsrsturi.streamlit.app/
- **User Guide:** [USER_GUIDE.md](USER_GUIDE.md)
- **Project Repository:** https://github.com/Julesvanh8/Data-engineering
- **Local Dashboard:** http://localhost:8501 (when running)
- **DBT Docs:** http://localhost:8080 (when running) 🌐 **Live Dashboard:** https://data-engineering-6urgspnpbupgepmsrsturi.streamlit.app/
- 📘 **User Guide (For Instructors):** [USER_GUIDE.md](USER_GUIDE.md)
- 📊 **GitHub Repository:** https://github.com/Julesvanh8/Data-engineering

---

## 🏗️ Modern Architecture

This project follows a clean, modular **Raw → SQL → DBT → Analysis** architecture:

```
Raw Data (APIs) → SQLite → DBT Transformations → Analysis/Dashboard
```

### Key Technologies
- **Python 3.12** - Data ingestion and analysis
- **SQLite** - Local data warehouse
- **DBT (dbt-sqlite)** - SQL-based transformations
- **Streamlit + Plotly** - Interactive dashboard
- **Pandas + NumPy** - Data manipulation
- **Matplotlib + Seaborn** - Static visualizations

---

## 📁 Project Structure

```
Data-engineering/
├── data/
│   ├── raw/                    # Raw data + SQLite databases
│   │   ├── market_data.db      # Source data
│   │   ├── main_staging.db     # DBT staging models
│   │   ├── main_intermediate.db # DBT intermediate models
│   │   └── main_marts.db       # DBT final marts (analytics-ready)
│   └── processed/              # Analysis outputs
│       └── events_combined.csv
│
├── src/
│   ├── 00_ingest/             # Data ingestion layer
│   │   ├── ingest_sp500.py         # S&P 500 from GitHub
│   │   ├── ingest_unemployment.py  # UNRATE from FRED
│   │   ├── ingest_tax_revenue.py   # Tax data from FRED
│   │   ├── db_utils.py             # Database utilities
│   │   └── run_ingest.py           # Master ingest script
│   │
│   ├── 01_transform/          # DBT transformation layer
│   │   └── transform/              # DBT project
│   │       ├── models/
│   │       │   ├── staging/        # stg_* models
│   │       │   ├── intermediate/   # int_* models
│   │       │   └── marts/          # fct_* models
│   │       └── dbt_project.yml
│   │
│   ├── 02_analysis/           # Analysis layer
│   │   ├── analyze_lags.py         # Event-study lag analysis
│   │   └── visualize.py            # Static visualizations
│   │
│   ├── 03_dashboard/          # Presentation layer
│   │   └── dashboard.py            # Interactive Streamlit dashboard
│   │
│   └── orchestration/         # Pipeline orchestration
│       └── run_pipeline.py         # Master pipeline runner
│
├── outputs/
│   ├── figures/               # Generated visualizations
│   └── tables/                # Analysis result tables
│
├── outputs/
│   ├── figures/               # Generated visualizations
│   └── tables/                # Analysis result tables
│
├── .env.example               # Environment variables template
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/Julesvanh8/Data-engineering.git
cd Data-engineering

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API key:

```bash
cp .env.example .env
# Edit .env and add your FRED API key
```

Get your free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html

### 3. Run the Complete Pipeline

```bash
# Run everything (ingest → transform → analyze)
python src/orchestration/run_pipeline.py

# Or run specific phases:
python src/orchestration/run_pipeline.py --skip-ingest      # Use existing data
python src/orchestration/run_pipeline.py --skip-analysis    # Just data prep
```

### 4. Launch Interactive Dashboard

```bash
streamlit run src/03_dashboard/dashboard.py
```

Opens at http://localhost:8501 with interactive visualizations.

---

## 📊 Data Sources

| Source | Dataset | Frequency | Time Range |
|--------|---------|-----------|------------|
| GitHub | S&P 500 Historical Data | Daily | 1871 - Present |
| FRED | UNRATE (Unemployment Rate) | Monthly | 1948 - Present |
| FRED | W006RC1Q027SBEA (Federal Tax) | Quarterly SAAR | 1947 - Present |

---

## 🔄 Pipeline Stages

### Stage 1: Data Ingestion (`src/00_ingest/`)
- Fetches S&P 500 data from GitHub datasets repository
- Fetches unemployment (UNRATE) from FRED API
- Fetches federal income tax revenue from FRED API
- Stores raw data in SQLite (`market_data.db`)
- Includes CSV backups in `data/raw/`

**Run manually:**
```bash
python src/00_ingest/run_ingest.py
```

### Stage 2: DBT Transformations (`src/01_transform/`)

DBT handles all data transformations using SQL:

**Staging Layer** (`models/staging/`):
- `stg_sp500` - Clean S&P 500 data
- `stg_unemployment` - Clean unemployment data
- `stg_tax_revenue` - Clean tax revenue data

**Intermediate Layer** (`models/intermediate/`):
- `int_monthly_returns` - Calculate monthly returns & downturn flags
- `int_tax_monthly` - Convert quarterly tax data to monthly (forward-fill)
- `int_lag_features` - Generate 1-12 month lag features

**Marts Layer** (`models/marts/`):
- `fct_combined_monthly` - Final analytics table combining all data

**Run manually:**
```bash
cd src/01_transform/transform
dbt run        # Execute all models
dbt test       # Run data quality tests
dbt docs generate && dbt docs serve  # View documentation
```

### Stage 3: Analysis (`src/02_analysis/`)

**Event Detection:**
- Identifies bear markets (≥19% drop from all-time high)
- Detects unemployment rises (≥2pp increase)
- Tracks tax revenue declines (≥7.5% drop)

**Lag Analysis:**
- Event-study methodology
- Tracks changes at 1-23 month lags
- Compares to pre-event baseline

**Run manually:**
```bash
python src/02_analysis/analyze_lags.py  # Generate event analysis
python src/02_analysis/visualize.py     # Create static plots
```

### Stage 4: Dashboard (`src/03_dashboard/`)

Interactive Streamlit dashboard with:
- Time series visualization with event shading
- Event study impulse-response charts
- Event catalog with detailed statistics
- Single event deep-dive analysis

---

## 📈 Key Findings

Based on event-study analysis of 9 bear market periods (1961-2021):

- **Unemployment rises** on average **2 months** after downturn starts
- Peak unemployment impact occurs at **23 months**
- Tax revenue shows long-term growth trend (SAAR data)
- Named events analyzed: Dot-com crash, Global Financial Crisis, COVID crash

For detailed results, see `data/processed/events_combined.csv` or launch the dashboard.

---

## 🧪 Testing & Validation

```bash
# Run DBT tests
cd src/01_transform/transform
dbt test

# Check data quality
python -c "
import sqlite3
conn = sqlite3.connect('data/raw/main_marts.db')
print(f'Rows in final mart: {conn.execute(\"SELECT COUNT(*) FROM fct_combined_monthly\").fetchone()[0]}')
conn.close()
"
```

---

## 🛠️ Development

### Adding New Data Sources

1. Create ingest script in `src/00_ingest/`
2. Add staging model in `src/01_transform/transform/models/staging/`
3. Update intermediate/marts models to include new data
4. Run `dbt run` to rebuild models

### Modifying Transformations

All business logic is in SQL within `src/01_transform/transform/models/`.

Edit `.sql` files and run:
```bash
dbt run --select model_name+  # Run model and downstream
```

### Adding Analysis

Create new scripts in `src/02_analysis/` that read from `main_marts.db`:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/raw/main_marts.db')
df = pd.read_sql('SELECT * FROM fct_combined_monthly', conn)
# Your analysis here...
```

---

## 📚 Documentation

- **DBT Models**: Run `dbt docs generate && dbt docs serve` for interactive docs
- **API Documentation**:
  - FRED API: https://fred.stlouisfed.org/docs/api/
  - S&P 500 Data: https://github.com/datasets/s-and-p-500

---

## 🤝 Contributing

This project follows a clean architecture:
- **Keep ingest separate from transformation**
- **Use SQL (DBT) for transformations, not Pandas/PySpark**
- **One source of truth**: DBT marts
- **Analysis scripts read from marts only**

---

## 👥 Team

- Jules van Halder
- Valerie
- Mohamed

---

## �� Links

- Project Repository: https://github.com/Julesvanh8/Data-engineering
- Dashboard (when running): http://localhost:8501
- DBT Docs (when running): http://localhost:8080
