# Data Engineering Project

## Research Question
How long after a U.S. stock market downturn do unemployment and federal income tax revenues change?

## Data Sources
- Alpha Vantage
- FRED
- World Bank

## Project Structure
- `data/raw/` for raw downloaded data
- `data/processed/` for cleaned and merged data
- `notebooks/` for exploratory analysis
- `src/` for Python scripts
- `outputs/figures/` for plots
- `outputs/tables/` for result tables

## Planned Workflow
1. Collect data from APIs
2. Clean and preprocess data
3. Merge datasets
4. Create downturn and lag variables
5. Analyze lagged effects
6. Export figures and results
7. Present the pipeline and conclusions

## API Keys Invullen
Maak in de projectroot een bestand `.env` en voeg deze regels toe:

```env
ALPHA_VANTAGE_API_KEY="Type hier je API key"
FRED_API_KEY="Type hier je API key"
```

Vervang daarna de placeholders door je echte sleutels.

## Datapipeline Uitvoeren
Het script staat in `src/fetch_prepare_pipeline.py` en doet het volgende:
- Haalt dagelijkse koersdata op voor `SPY` (of een andere ticker) via Alpha Vantage
- Valt automatisch terug op FRED `SP500` als Alpha Vantage beperkt is (rate-limit/premium)
- In `--stock-source auto` schakelt de pipeline ook naar FRED als de Alpha-historiek te kort is voor lag-analyse
- Berekent maandrendementen en een `downturn`-indicator
- Haalt `UNRATE` en federal income tax revenues op via FRED
- Harmoniseert alles naar maandfrequentie
- Schrijft ruwe data naar `data/raw/` en de samengestelde analysetabel naar `data/processed/`

## Frequenties Per Bron
- Aandelenmarkt (Alpha Vantage of FRED SP500): dagelijks
- Werkloosheid (`UNRATE`, FRED): maandelijks
- Federal income tax revenues (`W006RC1Q027SBEA`, FRED): kwartaal

In de verwerkte dataset wordt alles geharmoniseerd naar maandfrequentie.

Run:

```bash
python src/fetch_prepare_pipeline.py
```

Belangrijkste optionele argumenten:

```bash
python src/fetch_prepare_pipeline.py --symbol SPY --downturn-threshold -0.10 --tax-series W006RC1Q027SBEA
```

Bron voor aandelenreeks kiezen:

```bash
python src/fetch_prepare_pipeline.py --stock-source auto
python src/fetch_prepare_pipeline.py --stock-source fred --fred-stock-series SP500
python src/fetch_prepare_pipeline.py --stock-source alpha
```

Optioneel kan je een World Bank indicator toevoegen:

```bash
python src/fetch_prepare_pipeline.py --world-bank-indicator NY.GDP.MKTP.CD
```

## Verwachte Output
- Ruwe bestanden in `data/raw/`
- Geharmoniseerde maanddataset in `data/processed/merged_monthly_dataset.csv`
