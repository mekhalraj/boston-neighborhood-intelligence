# Boston Neighborhood Safety Map

Interactive safety dashboard for Boston neighborhoods. Combines crime incidents, 311 complaints, Vision Zero crash data, and building violations into composite safety scores across 23 neighborhoods.

**Live:** [mekhalraj.github.io/boston-neighborhood-intelligence](https://mekhalraj.github.io/boston-neighborhood-intelligence)

## Features

- Choropleth map colored by safety scores (MapLibre GL JS)
- Toggle between Overall, Crime, 311, Crashes, and Violations views
- Click any neighborhood for detailed breakdown with radar chart
- Compare two neighborhoods side by side
- Sortable ranking table
- Auto-generated data insights

## Data Sources

All data from [Analyze Boston](https://data.boston.gov):

- Crime Incident Reports (2023-2025)
- 311 Service Requests (2025)
- Vision Zero Crash Records (2023-2025)
- Building & Property Violations (2023-2025)

## Stack

- **Data pipeline:** Python (pandas, geopandas) → JSON files
- **Frontend:** Single HTML file (MapLibre GL JS + Chart.js)
- **Hosting:** GitHub Pages

## Running the Pipeline

```bash
pip install -r requirements.txt
python pipeline/build_data.py
```

## Local Development

```bash
python -m http.server 8000
# Open http://localhost:8000
```

---

Phase 1 of Boston Neighborhood Intelligence. Built by [Mekhal Raj](https://www.linkedin.com/in/mekhalraj/).
