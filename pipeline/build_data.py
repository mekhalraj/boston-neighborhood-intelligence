"""
Boston Neighborhood Safety Map -- Data Pipeline
Downloads data from Analyze Boston, processes it, and exports JSON files.
"""

import json
import os
import sys
import time
from contextlib import contextmanager
from io import StringIO

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

CKAN_BASE = "https://data.boston.gov/api/3/action"

RESOURCES = {
    "311": [
        "e6013a93-1321-4f2a-bf91-8d8a02f1e62f",  # 2023
        "dff4d804-5031-443a-8409-8344efd0e5c8",  # 2024
        "9d7c2214-4709-478a-a2e8-fb2020a5bb94",  # 2025
    ],
    "crime": "b973d8cb-eeb2-4e7e-99da-c92938efc9c0",
    "violations": "800a2663-1d6a-46e7-9356-bedb70f5332c",
    "crashes": "e4bfe397-6bfc-49c5-9367-c879fac7401d",
}

NEIGHBORHOODS_GEOJSON_URL = (
    "https://data.boston.gov/dataset/boston-neighborhood-boundaries-approximated-by-2020-census-tracts/"
    "resource/42a271c9-486d-4f9e-adc2-63e4bf47fe3e/download/"
    "boston_neighborhood_boundaries_approximated_by_2020_census_tracts.geojson"
)

# BPDA Research Division, January 2025 estimates (primary)
# ACS/Census estimates for sub-areas not in BPDA data (supplementary)
# Source: Boston in Context: Neighborhoods (bostonplans.org)
POPULATION = {
    # BPDA 2025 neighborhoods
    "Allston": 31810,
    "Back Bay": 18983,
    "Beacon Hill": 9327,
    "Brighton": 55869,
    "Charlestown": 19232,
    "Chinatown": 6371,
    "Dorchester": 123056,
    "Downtown": 15752,
    "East Boston": 46892,
    "Fenway": 42351,
    "Hyde Park": 33469,
    "Jamaica Plain": 40083,
    "Longwood": 5765,
    "Mattapan": 26796,
    "Mission Hill": 19076,
    "North End": 10080,
    "Roslindale": 35479,
    "Roxbury": 53821,
    "South Boston": 38263,
    "South End": 35287,
    "West End": 5014,
    "West Roxbury": 33906,
    # Additional neighborhoods from GeoJSON (ACS/Census estimates)
    "South Boston Waterfront": 5300,
}

# Weights for overall safety score
WEIGHTS = {"crime": 0.40, "complaints": 0.25, "crashes": 0.20, "violations": 0.15}

SMALL_POP_THRESHOLD = 5000

# Year filter -- use most recent 2 full years of data
YEAR_MIN = 2023
YEAR_MAX = 2025


# ---------------------------------------------------------------------------
# Neighborhood name normalization
# ---------------------------------------------------------------------------

# Maps variant names found in datasets to canonical names matching POPULATION keys
# With 34 individual neighborhoods, most GeoJSON names are canonical.
# Only map 311/crime data variants that don't match GeoJSON names.
NAME_ALIASES = {
    "Downtown / Financial District": "Downtown",
    "Seaport": "South Boston Waterfront",
    "South Boston / South Boston Waterfront": "South Boston",
    "Longwood Medical and Academic Area": "Longwood",
    "Longwood Medical Area": "Longwood",
    "Harbor Islands": None,  # exclude -- no residential population
    "Harbor Island": None,
}


# ---------------------------------------------------------------------------
# Cambridge configuration
# ---------------------------------------------------------------------------

CAM_SOCRATA_BASE = "https://data.cambridgema.gov/resource"

CAM_RESOURCES = {
    "crime":   "xuad-73uj",   # Crime Reports (2009-present)
    "311":     "2z9k-mv9g",   # Commonwealth Connect Service Requests (SeeClickFix)
    "crashes": "ybny-g9cv",   # Police Department Crash Data - Updated
    "permits": "qu2z-8suj",   # Building Permits: Addition/Alteration
}

CAM_NEIGHBORHOODS_URLS = [
    "https://raw.githubusercontent.com/cambridgegis/cambridgegis_data/main/Boundary/CDD_Neighborhoods/BOUNDARY_CDDNeighborhoods.geojson",
    "https://data.cambridgema.gov/api/geospatial/2iqn-k6m9?method=export&type=GeoJSON",
]

# Cambridge CDD defines 13 neighborhoods; populations from 2020 Census / MAPC estimates
# "Agassiz" is part of Neighborhood Nine in the CDD classification
# "The Port" and "Baldwin" are distinct CDD neighborhoods
CAM_POPULATION = {
    "Area 2/MIT":           7200,
    "Baldwin":              8900,
    "Cambridge Highlands":  2100,
    "Cambridgeport":       11300,
    "East Cambridge":      11800,
    "Mid-Cambridge":       16400,
    "Neighborhood Nine":    5800,
    "North Cambridge":     16200,
    "Riverside":            8400,
    "Strawberry Hill":      3800,
    "The Port":             9700,
    "Wellington-Harrington": 9700,
    "West Cambridge":       8100,
}

CAM_NAME_ALIASES = {
    "Area 2/ MIT": "Area 2/MIT",
    "Area 2 / MIT": "Area 2/MIT",
    "Area II/MIT": "Area 2/MIT",
    "Area II / MIT": "Area 2/MIT",
    "Wellington Harrington": "Wellington-Harrington",
    "Wellington - Harrington": "Wellington-Harrington",
    "Neighborhood 9": "Neighborhood Nine",
}

CAM_DATA_DIR = os.path.join(DATA_DIR, "cambridge")


@contextmanager
def _cambridge_context():
    """Temporarily swap globals so reused functions see Cambridge data."""
    global POPULATION, NAME_ALIASES
    orig_pop, orig_aliases = POPULATION, NAME_ALIASES
    POPULATION, NAME_ALIASES = CAM_POPULATION, CAM_NAME_ALIASES
    try:
        yield
    finally:
        POPULATION, NAME_ALIASES = orig_pop, orig_aliases


def normalize_neighborhood(name):
    """Normalize a neighborhood name to our canonical set."""
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    # Check alias map first
    if name in NAME_ALIASES:
        return NAME_ALIASES[name]
    # Check if it's already a canonical name
    if name in POPULATION:
        return name
    # Try title case
    title = name.strip().title()
    if title in POPULATION:
        return title
    return name  # return as-is; will be logged if not in POPULATION


# ---------------------------------------------------------------------------
# Data download functions
# ---------------------------------------------------------------------------

def download_datastore(resource_id, label, limit=32000):
    """Download a dataset from the CKAN datastore API with pagination."""
    all_records = []
    offset = 0
    while True:
        for attempt in range(3):
            try:
                print(f"  Downloading {label}: offset={offset}...")
                r = requests.get(
                    f"{CKAN_BASE}/datastore_search",
                    params={"resource_id": resource_id, "limit": limit, "offset": offset},
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  Retry {attempt+1} for {label}: {e}")
                time.sleep(2 ** attempt)

        records = data["result"]["records"]
        if not records:
            break
        all_records.extend(records)
        print(f"  ... {len(all_records)} records so far")
        if len(records) < limit:
            break
        offset += limit

    print(f"  {label}: {len(all_records)} total records downloaded")
    return pd.DataFrame(all_records)


def download_csv_direct(resource_id, label):
    """Download a dataset as CSV directly (fallback for large files)."""
    # First get the resource URL
    print(f"  Getting resource URL for {label}...")
    r = requests.get(
        f"{CKAN_BASE}/resource_show",
        params={"id": resource_id},
        timeout=30,
    )
    r.raise_for_status()
    url = r.json()["result"]["url"]

    print(f"  Downloading CSV from {url}...")
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    df = pd.read_csv(StringIO(r.text), low_memory=False)
    print(f"  {label}: {len(df)} records downloaded")
    return df


def download_geojson():
    """Download Boston neighborhood boundaries GeoJSON."""
    print("Downloading neighborhood boundaries...")
    for attempt in range(3):
        try:
            r = requests.get(NEIGHBORHOODS_GEOJSON_URL, timeout=60)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
    # Keep original name for reference, add canonical name
    print(f"  Neighborhoods found: {sorted(gdf['neighborhood'].unique())}")
    gdf["canonical_name"] = gdf["neighborhood"].apply(normalize_neighborhood)
    gdf = gdf[gdf["canonical_name"].notna()].copy()
    print(f"  Canonical neighborhoods: {sorted(gdf['canonical_name'].unique())}")
    return gdf, r.json()


def download_socrata(dataset_id, label, limit=50000):
    """Download a dataset from Cambridge's Socrata SODA API with pagination."""
    all_records = []
    offset = 0
    headers = {"User-Agent": "BostonNeighborhoodIntelligence/1.0"}
    while True:
        url = f"{CAM_SOCRATA_BASE}/{dataset_id}.json"
        params = {"$limit": limit, "$offset": offset}
        for attempt in range(3):
            try:
                print(f"  Downloading {label}: offset={offset}...")
                r = requests.get(url, params=params, headers=headers, timeout=120)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  Retry {attempt+1} for {label}: {e}")
                time.sleep(2 ** attempt)

        # Socrata may return an error object instead of an array
        if isinstance(data, dict) and ("error" in data or "message" in data):
            raise RuntimeError(f"Socrata API error for {label}: {data}")

        if not data:
            break
        all_records.extend(data)
        print(f"  ... {len(all_records)} records so far")
        if len(data) < limit:
            break
        offset += limit
        time.sleep(0.3)  # be polite to Socrata

    print(f"  {label}: {len(all_records)} total records downloaded")
    return pd.DataFrame(all_records)


def download_cambridge_geojson():
    """Download Cambridge neighborhood boundaries GeoJSON."""
    print("Downloading Cambridge neighborhood boundaries...")
    raw = None
    for url in CAM_NEIGHBORHOODS_URLS:
        for attempt in range(3):
            try:
                print(f"  Trying {url[:80]}...")
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                raw = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  Failed: {e}")
                else:
                    print(f"  Retry {attempt+1}: {e}")
                    time.sleep(2 ** attempt)
        if raw and "features" in raw and len(raw["features"]) > 0:
            break
        raw = None

    if not raw:
        raise RuntimeError("Could not download Cambridge neighborhood GeoJSON from any URL")

    gdf = gpd.GeoDataFrame.from_features(raw["features"], crs="EPSG:4326")

    # Find the neighborhood name column (varies by source)
    name_col = None
    for candidate in ["NAME", "Name", "name", "NHOOD", "neighborhood", "Neighborhood", "NBHD"]:
        if candidate in gdf.columns:
            name_col = candidate
            break
    if name_col is None:
        # Fall back to first string column
        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype == object:
                name_col = col
                break
    if name_col is None:
        raise RuntimeError(f"Cannot find neighborhood name column. Columns: {list(gdf.columns)}")

    gdf["neighborhood"] = gdf[name_col].astype(str).str.strip()
    print(f"  Cambridge neighborhoods found: {sorted(gdf['neighborhood'].unique())}")

    gdf["canonical_name"] = gdf["neighborhood"].apply(normalize_neighborhood)
    gdf = gdf[gdf["canonical_name"].notna()].copy()
    print(f"  Canonical Cambridge neighborhoods: {sorted(gdf['canonical_name'].unique())}")

    # Ensure raw GeoJSON features have "neighborhood" property for enrich_geojson
    for feature in raw["features"]:
        props = feature.get("properties", {})
        if "neighborhood" not in props:
            for candidate in ["NAME", "Name", "name", "NHOOD", "Neighborhood", "NBHD"]:
                if candidate in props:
                    feature["properties"]["neighborhood"] = props[candidate]
                    break

    return gdf, raw


# ---------------------------------------------------------------------------
# Data cleaning functions
# ---------------------------------------------------------------------------

def clean_311(df):
    """Clean 311 service requests."""
    print("Cleaning 311 data...")
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()

    # Parse dates
    for col in ["open_date", "open_dt"]:
        if col in df.columns:
            df["open_date_parsed"] = pd.to_datetime(df[col], errors="coerce")
            break

    for col in ["close_date", "closed_dt", "closed_date"]:
        if col in df.columns:
            df["close_date_parsed"] = pd.to_datetime(df[col], errors="coerce")
            break

    if "open_date_parsed" not in df.columns:
        print("  WARNING: Could not find open_date column!")
        print(f"  Available columns: {list(df.columns)}")
        return pd.DataFrame()

    df["year"] = df["open_date_parsed"].dt.year
    df["month"] = df["open_date_parsed"].dt.month
    df["day_of_week"] = df["open_date_parsed"].dt.day_name()
    df["hour"] = df["open_date_parsed"].dt.hour

    # Filter years
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

    # Get category field
    for col in ["case_topic", "reason", "type", "service_name"]:
        if col in df.columns:
            df["category"] = df[col].fillna("Unknown")
            break
    if "category" not in df.columns:
        df["category"] = "Unknown"

    # Normalize neighborhood field if it exists
    if "neighborhood" in df.columns:
        df["neighborhood_orig"] = df["neighborhood"]
        df["neighborhood"] = df["neighborhood"].apply(normalize_neighborhood)

    # Compute response time in hours
    if "close_date_parsed" in df.columns:
        df["response_hours"] = (
            df["close_date_parsed"] - df["open_date_parsed"]
        ).dt.total_seconds() / 3600
        # Drop negative or extreme values (> 365 days)
        df.loc[df["response_hours"] < 0, "response_hours"] = np.nan
        df.loc[df["response_hours"] > 8760, "response_hours"] = np.nan

    # Convert lat/lon to float
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  311 after cleaning: {len(df)} records, years {df['year'].min()}-{df['year'].max()}")
    return df


def clean_crime(df):
    """Clean crime incident reports."""
    print("Cleaning crime data...")
    df.columns = df.columns.str.strip()

    # Handle case variations in column names
    col_map = {}
    for c in df.columns:
        col_map[c.upper()] = c

    # Parse date
    date_col = col_map.get("OCCURRED_ON_DATE", "OCCURRED_ON_DATE")
    if date_col in df.columns:
        df["date_parsed"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        print(f"  WARNING: Could not find OCCURRED_ON_DATE. Columns: {list(df.columns)}")
        return pd.DataFrame()

    # Year filter
    year_col = col_map.get("YEAR", "YEAR")
    if year_col in df.columns:
        df["year"] = pd.to_numeric(df[year_col], errors="coerce")
    else:
        df["year"] = df["date_parsed"].dt.year

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

    # Hour and day
    hour_col = col_map.get("HOUR", "HOUR")
    if hour_col in df.columns:
        df["hour"] = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int)
    else:
        df["hour"] = df["date_parsed"].dt.hour

    dow_col = col_map.get("DAY_OF_WEEK", "DAY_OF_WEEK")
    if dow_col in df.columns:
        df["day_of_week"] = df[dow_col]
    else:
        df["day_of_week"] = df["date_parsed"].dt.day_name()

    df["month"] = df["date_parsed"].dt.month

    # Offense group
    offense_col = col_map.get("OFFENSE_CODE_GROUP", "OFFENSE_CODE_GROUP")
    if offense_col in df.columns:
        df["offense_group"] = df[offense_col].fillna("Other")
    else:
        desc_col = col_map.get("OFFENSE_DESCRIPTION", "OFFENSE_DESCRIPTION")
        if desc_col in df.columns:
            df["offense_group"] = df[desc_col].fillna("Other")
        else:
            df["offense_group"] = "Other"

    # Shooting
    shooting_col = col_map.get("SHOOTING", "SHOOTING")
    if shooting_col in df.columns:
        df["shooting"] = df[shooting_col].astype(str).str.upper().str.strip() == "Y"
    else:
        df["shooting"] = False

    # Lat/Long
    lat_col = col_map.get("LAT", "Lat")
    long_col = col_map.get("LONG", "Long")
    for c in ["Lat", "lat", "LAT"]:
        if c in df.columns:
            lat_col = c
            break
    for c in ["Long", "long", "LONG"]:
        if c in df.columns:
            long_col = c
            break

    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col in df.columns else np.nan
    df["lon"] = pd.to_numeric(df[long_col], errors="coerce") if long_col in df.columns else np.nan

    # Drop missing/zero coords
    df = df[(df["lat"].notna()) & (df["lon"].notna())].copy()
    df = df[(df["lat"] != 0) & (df["lon"] != 0)].copy()
    # Filter to reasonable Boston bounds
    df = df[(df["lat"] > 42.2) & (df["lat"] < 42.5) & (df["lon"] > -71.2) & (df["lon"] < -70.9)].copy()

    print(f"  Crime after cleaning: {len(df)} records, years {df['year'].min()}-{df['year'].max()}")
    return df


def clean_violations(df):
    """Clean building & property violations."""
    print("Cleaning violations data...")
    df.columns = df.columns.str.lower().str.strip()

    # Parse date
    for col in ["status_dttm", "statusdttm", "date"]:
        if col in df.columns:
            df["date_parsed"] = pd.to_datetime(df[col], errors="coerce")
            break

    if "date_parsed" in df.columns:
        df["year"] = df["date_parsed"].dt.year
        df["month"] = df["date_parsed"].dt.month
        df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    else:
        print(f"  WARNING: No date column found. Columns: {list(df.columns)}")
        # Keep all records if no date

    # Coords
    df["latitude"] = pd.to_numeric(df.get("latitude", pd.Series(dtype=float)), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude", pd.Series(dtype=float)), errors="coerce")
    df = df[(df["latitude"].notna()) & (df["longitude"].notna())].copy()
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()

    # Description
    if "description" in df.columns:
        df["violation_desc"] = df["description"].fillna("Unknown")
    elif "value" in df.columns:
        df["violation_desc"] = df["value"].fillna("Unknown")
    else:
        df["violation_desc"] = "Unknown"

    print(f"  Violations after cleaning: {len(df)} records")
    return df


def clean_crashes(df):
    """Clean Vision Zero crash records."""
    print("Cleaning crash data...")
    df.columns = df.columns.str.lower().str.strip()

    # Parse date
    if "dispatch_ts" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["dispatch_ts"], errors="coerce", utc=True)
        df["date_parsed"] = df["date_parsed"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    else:
        print(f"  WARNING: No dispatch_ts column. Columns: {list(df.columns)}")
        return pd.DataFrame()

    df["year"] = df["date_parsed"].dt.year
    df["month"] = df["date_parsed"].dt.month
    df["hour"] = df["date_parsed"].dt.hour
    df["day_of_week"] = df["date_parsed"].dt.day_name()

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

    # Mode type
    if "mode_type" in df.columns:
        df["mode"] = df["mode_type"].fillna("unknown").str.lower().str.strip()
    else:
        df["mode"] = "unknown"

    # Coords
    df["lat"] = pd.to_numeric(df.get("lat", pd.Series(dtype=float)), errors="coerce")
    df["long"] = pd.to_numeric(df.get("long", pd.Series(dtype=float)), errors="coerce")
    df = df[(df["lat"].notna()) & (df["long"].notna())].copy()
    df = df[(df["lat"] != 0) & (df["long"] != 0)].copy()

    print(f"  Crashes after cleaning: {len(df)} records, years {df['year'].min()}-{df['year'].max()}")
    return df


# ---------------------------------------------------------------------------
# Cambridge data cleaning functions
# ---------------------------------------------------------------------------

def _extract_location_coords(df, lat_col_names=None, lon_col_names=None):
    """Extract lat/lon from Socrata location dict or flat columns."""
    lat_col_names = lat_col_names or ["latitude", "lat"]
    lon_col_names = lon_col_names or ["longitude", "lng", "lon", "long"]

    df = df.copy()
    df["_lat"] = np.nan
    df["_lon"] = np.nan

    # Try flat columns first
    for col in lat_col_names:
        if col in df.columns:
            df["_lat"] = pd.to_numeric(df[col], errors="coerce")
            break
    for col in lon_col_names:
        if col in df.columns:
            df["_lon"] = pd.to_numeric(df[col], errors="coerce")
            break

    # Fall back to location dict
    if "location" in df.columns and df["_lat"].isna().all():
        def parse_loc(val):
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                try:
                    import ast
                    return ast.literal_eval(val)
                except Exception:
                    return {}
            return {}

        locs = df["location"].apply(parse_loc)
        df["_lat"] = pd.to_numeric(locs.apply(lambda d: d.get("latitude")), errors="coerce")
        df["_lon"] = pd.to_numeric(locs.apply(lambda d: d.get("longitude")), errors="coerce")

    return df


def clean_cambridge_crime(df):
    """Clean Cambridge crime reports from Socrata."""
    print("Cleaning Cambridge crime data...")
    df.columns = df.columns.str.lower().str.strip()
    print(f"  Crime columns: {list(df.columns)[:15]}")

    # Parse date -- Cambridge uses date_of_report or crime_date_time
    date_parsed = None
    for col in ["date_of_report", "crime_date_time", "occurred_on_date", "date"]:
        if col in df.columns:
            date_parsed = pd.to_datetime(df[col], errors="coerce")
            break
    if date_parsed is None:
        print(f"  WARNING: No date column found. Columns: {list(df.columns)}")
        return pd.DataFrame()
    df["date_parsed"] = date_parsed

    df["year"] = df["date_parsed"].dt.year
    df["month"] = df["date_parsed"].dt.month
    df["hour"] = df["date_parsed"].dt.hour
    df["day_of_week"] = df["date_parsed"].dt.day_name()

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

    # Offense group -- Cambridge uses "crime" column
    for col in ["crime", "offense", "crime_type", "offense_description"]:
        if col in df.columns:
            df["offense_group"] = df[col].fillna("Other")
            break
    if "offense_group" not in df.columns:
        df["offense_group"] = "Other"

    df["shooting"] = False

    # Coordinates -- Cambridge uses reporting_area_lat/reporting_area_lon
    df = _extract_location_coords(
        df,
        lat_col_names=["reporting_area_lat", "latitude", "lat"],
        lon_col_names=["reporting_area_lon", "longitude", "lng", "lon", "long"],
    )
    df["lat"] = df["_lat"]
    df["lon"] = df["_lon"]
    df = df.drop(columns=["_lat", "_lon"])
    df = df[(df["lat"].notna()) & (df["lon"].notna())].copy()
    df = df[(df["lat"] != 0) & (df["lon"] != 0)].copy()
    # Cambridge bounds
    df = df[(df["lat"] > 42.34) & (df["lat"] < 42.42) & (df["lon"] > -71.17) & (df["lon"] < -71.05)].copy()

    # Cambridge crime data has a neighborhood column -- normalize it
    if "neighborhood" in df.columns:
        df["neighborhood"] = df["neighborhood"].apply(normalize_neighborhood)

    print(f"  Cambridge crime after cleaning: {len(df)} records")
    return df


def clean_cambridge_311(df):
    """Clean Cambridge 311/SeeClickFix requests from Socrata."""
    print("Cleaning Cambridge 311 data...")
    df.columns = df.columns.str.lower().str.strip()

    # Parse open date -- Cambridge uses ticket_created_date_time
    for col in ["ticket_created_date_time", "created", "requested_datetime", "open_date", "open_dt"]:
        if col in df.columns:
            df["open_date_parsed"] = pd.to_datetime(df[col], errors="coerce")
            break
    if "open_date_parsed" not in df.columns:
        print(f"  WARNING: No date column found. Columns: {list(df.columns)}")
        return pd.DataFrame()

    df["year"] = df["open_date_parsed"].dt.year
    df["month"] = df["open_date_parsed"].dt.month
    df["day_of_week"] = df["open_date_parsed"].dt.day_name()
    df["hour"] = df["open_date_parsed"].dt.hour

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

    # Category -- Cambridge uses issue_type
    for col in ["issue_type", "service_name", "category", "department"]:
        if col in df.columns:
            df["category"] = df[col].fillna("Unknown")
            break
    if "category" not in df.columns:
        df["category"] = "Unknown"

    # Close date and response time -- Cambridge uses ticket_last_updated_date_time
    for col in ["ticket_last_updated_date_time", "updated_datetime", "closed_datetime", "close_date", "closed_dt"]:
        if col in df.columns:
            df["close_date_parsed"] = pd.to_datetime(df[col], errors="coerce")
            break
    if "close_date_parsed" in df.columns:
        # Only compute for closed/resolved requests
        status_col = None
        for col in ["ticket_status", "status"]:
            if col in df.columns:
                status_col = col
                break
        if status_col:
            closed_mask = df[status_col].astype(str).str.lower().str.contains("closed|resolved|completed|archived", na=False)
            df.loc[~closed_mask, "close_date_parsed"] = pd.NaT
        df["response_hours"] = (
            df["close_date_parsed"] - df["open_date_parsed"]
        ).dt.total_seconds() / 3600
        df.loc[df["response_hours"] < 0, "response_hours"] = np.nan
        df.loc[df["response_hours"] > 8760, "response_hours"] = np.nan

    # Coordinates -- Cambridge uses lat/lng
    df = _extract_location_coords(df, lat_col_names=["lat", "latitude"], lon_col_names=["lng", "longitude", "lon", "long"])
    df["latitude"] = df["_lat"]
    df["longitude"] = df["_lon"]
    df = df.drop(columns=["_lat", "_lon"])

    # Use existing neighborhood field if present, otherwise create empty one
    if "neighborhood" in df.columns:
        df["neighborhood_orig"] = df["neighborhood"]
        df["neighborhood"] = df["neighborhood"].apply(normalize_neighborhood)
    else:
        df["neighborhood"] = np.nan

    print(f"  Cambridge 311 after cleaning: {len(df)} records")
    return df


def clean_cambridge_crashes(df):
    """Clean Cambridge crash data from Socrata."""
    print("Cleaning Cambridge crash data...")
    df.columns = df.columns.str.lower().str.strip()

    # Parse date
    date_parsed = None
    for col in ["date_time", "date_of_crash", "crash_date_time", "date"]:
        if col in df.columns:
            date_parsed = pd.to_datetime(df[col], errors="coerce")
            break
    if date_parsed is None:
        print(f"  WARNING: No date column found. Columns: {list(df.columns)}")
        return pd.DataFrame()
    df["date_parsed"] = date_parsed

    df["year"] = df["date_parsed"].dt.year
    df["month"] = df["date_parsed"].dt.month
    df["hour"] = df["date_parsed"].dt.hour
    df["day_of_week"] = df["date_parsed"].dt.day_name()

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

    # Mode detection: Cambridge uses object_1/object_2 columns (e.g. "Pedestrian", "Bicycle")
    df["mode"] = "mv"
    # Check object_1 and object_2 for pedestrian/bicycle involvement
    for col in ["object_1", "object_2"]:
        if col in df.columns:
            vals = df[col].fillna("").str.lower()
            df.loc[vals.str.contains("ped"), "mode"] = "ped"
            df.loc[vals.str.contains("bic|cycl"), "mode"] = "bike"
    # Also check for boolean/flag columns
    ped_cols = [c for c in df.columns if "pedestrian" in c.lower() or "ped" == c.lower()]
    bike_cols = [c for c in df.columns if "bicycle" in c.lower() or "cyclist" in c.lower() or "bike" == c.lower()]
    for col in ped_cols:
        mask = df[col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        df.loc[mask, "mode"] = "ped"
    for col in bike_cols:
        mask = df[col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        df.loc[mask, "mode"] = "bike"
    # Also check a single mode_type column
    if "mode_type" in df.columns:
        mt = df["mode_type"].fillna("").str.lower().str.strip()
        df.loc[mt.str.contains("ped"), "mode"] = "ped"
        df.loc[mt.str.contains("bik|cycl"), "mode"] = "bike"

    # Coordinates
    df = _extract_location_coords(df)
    df["lat"] = df["_lat"]
    df["long"] = df["_lon"]
    df = df.drop(columns=["_lat", "_lon"])
    df = df[(df["lat"].notna()) & (df["long"].notna())].copy()
    df = df[(df["lat"] != 0) & (df["long"] != 0)].copy()

    print(f"  Cambridge crashes after cleaning: {len(df)} records")
    return df


def clean_cambridge_permits(df):
    """Clean Cambridge building permits (used as violations proxy) from Socrata."""
    print("Cleaning Cambridge permits data...")
    df.columns = df.columns.str.lower().str.strip()

    # Parse date
    for col in ["issue_date", "issued_date", "permit_date", "date_issued", "applicant_submit_date", "date"]:
        if col in df.columns:
            df["date_parsed"] = pd.to_datetime(df[col], errors="coerce")
            break
    if "date_parsed" in df.columns:
        df["year"] = df["date_parsed"].dt.year
        df["month"] = df["date_parsed"].dt.month
        df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    else:
        print(f"  WARNING: No date column found. Columns: {list(df.columns)}")

    # Coordinates
    df = _extract_location_coords(df)
    df["latitude"] = df["_lat"]
    df["longitude"] = df["_lon"]
    df = df.drop(columns=["_lat", "_lon"])
    df = df[(df["latitude"].notna()) & (df["longitude"].notna())].copy()
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()

    # Set violation_desc for compatibility with aggregate_violations
    df["violation_desc"] = "Building Permit"

    print(f"  Cambridge permits after cleaning: {len(df)} records")
    return df


# ---------------------------------------------------------------------------
# Spatial join
# ---------------------------------------------------------------------------

def spatial_join(df, neighborhoods_gdf, lat_col, lon_col):
    """Assign each record to a neighborhood via point-in-polygon."""
    print(f"  Spatial join: {len(df)} records...")
    df_reset = df.reset_index(drop=True).copy()
    # Drop any existing 'neighborhood' column to avoid conflicts with the join
    cols_to_drop = [c for c in df_reset.columns if c in ("canonical_name", "index_right", "neighborhood")]
    if cols_to_drop:
        df_reset = df_reset.drop(columns=cols_to_drop)
    points = gpd.GeoDataFrame(
        df_reset,
        geometry=gpd.points_from_xy(df_reset[lon_col], df_reset[lat_col]),
        crs="EPSG:4326",
    )
    hood_gdf = neighborhoods_gdf[["canonical_name", "geometry"]].copy()
    joined = gpd.sjoin(points, hood_gdf, how="left", predicate="within")
    # sjoin can create duplicates when points are on borders; keep first match
    joined = joined.loc[~joined.index.duplicated(keep="first")]
    joined = joined.rename(columns={"canonical_name": "neighborhood"})
    # Drop records that didn't match any neighborhood
    before = len(joined)
    joined = joined[joined["neighborhood"].notna()]
    after = len(joined)
    if before > after:
        print(f"  Dropped {before - after} records outside neighborhoods ({(before-after)/before*100:.1f}%)")
    result = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    return result


def assign_neighborhoods_311(df, neighborhoods_gdf):
    """For 311 data, use the existing neighborhood field when available, spatial join for the rest."""
    # Records with valid neighborhood field
    has_hood = df["neighborhood"].notna() & df["neighborhood"].isin(POPULATION.keys())
    df_with = df[has_hood].copy()

    # Records needing spatial join
    df_without = df[~has_hood].copy()
    if len(df_without) > 0 and "latitude" in df_without.columns:
        valid_coords = df_without["latitude"].notna() & df_without["longitude"].notna()
        df_need_join = df_without[valid_coords].copy()
        df_no_coords = df_without[~valid_coords].copy()

        if len(df_need_join) > 0:
            df_joined = spatial_join(df_need_join, neighborhoods_gdf, "latitude", "longitude")
            df_without = pd.concat([df_joined, df_no_coords], ignore_index=True)
        else:
            df_without = df_no_coords

    result = pd.concat([df_with, df_without], ignore_index=True)
    result = result[result["neighborhood"].notna() & result["neighborhood"].isin(POPULATION.keys())]
    print(f"  311 after neighborhood assignment: {len(result)} records")
    return result


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def aggregate_311(df):
    """Aggregate 311 data by neighborhood."""
    print("Aggregating 311 data...")
    result = {}

    for hood, group in df.groupby("neighborhood"):
        pop = POPULATION.get(hood, 0)
        total = len(group)

        # Top categories
        top_cats = (
            group["category"]
            .value_counts()
            .head(5)
            .reset_index()
            .rename(columns={"category": "topic", "count": "count"})
        )
        top_topics = [{"topic": r["topic"], "count": int(r["count"])} for _, r in top_cats.iterrows()]

        # Monthly counts
        by_month = {}
        for (y, m), mg in group.groupby(["year", "month"]):
            key = f"{int(y)}-{int(m):02d}"
            by_month[key] = len(mg)

        # Avg response time
        avg_response = None
        if "response_hours" in group.columns:
            valid_resp = group["response_hours"].dropna()
            if len(valid_resp) > 0:
                avg_response = round(float(valid_resp.median()), 1)  # median more robust

        result[hood] = {
            "total": total,
            "per_1000": round(total / pop * 1000, 1) if pop > 0 else 0,
            "top_topics": top_topics,
            "by_month": dict(sorted(by_month.items())),
            "avg_response_hours": avg_response,
        }

    return result


def aggregate_crime(df):
    """Aggregate crime data by neighborhood."""
    print("Aggregating crime data...")
    result = {}

    for hood, group in df.groupby("neighborhood"):
        pop = POPULATION.get(hood, 0)
        total = len(group)

        # Top offense groups
        top_off = (
            group["offense_group"]
            .value_counts()
            .head(5)
            .reset_index()
            .rename(columns={"offense_group": "offense", "count": "count"})
        )
        top_offenses = [{"offense": r["offense"], "count": int(r["count"])} for _, r in top_off.iterrows()]

        # By hour (24 bins)
        by_hour = [0] * 24
        for h, hg in group.groupby("hour"):
            h = int(h)
            if 0 <= h < 24:
                by_hour[h] = len(hg)

        # By day of week
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_day = {}
        day_counts = group["day_of_week"].value_counts()
        for d in day_order:
            by_day[d] = int(day_counts.get(d, 0))

        # By year
        by_year = {}
        for y, yg in group.groupby("year"):
            by_year[str(int(y))] = len(yg)

        # Shootings
        shootings = int(group["shooting"].sum()) if "shooting" in group.columns else 0

        result[hood] = {
            "total": total,
            "per_1000": round(total / pop * 1000, 1) if pop > 0 else 0,
            "top_offenses": top_offenses,
            "by_hour": by_hour,
            "by_day": by_day,
            "by_year": dict(sorted(by_year.items())),
            "shootings": shootings,
        }

    return result


def aggregate_crashes(df):
    """Aggregate crash data by neighborhood."""
    print("Aggregating crash data...")
    result = {}

    for hood, group in df.groupby("neighborhood"):
        pop = POPULATION.get(hood, 0)
        total = len(group)

        # By mode type
        mode_counts = group["mode"].value_counts()
        pedestrian = int(mode_counts.get("ped", 0))
        cyclist = int(mode_counts.get("bike", 0))
        mv = int(mode_counts.get("mv", 0))

        # By hour
        by_hour = [0] * 24
        for h, hg in group.groupby("hour"):
            h = int(h)
            if 0 <= h < 24:
                by_hour[h] = len(hg)

        # By year
        by_year = {}
        for y, yg in group.groupby("year"):
            by_year[str(int(y))] = len(yg)

        result[hood] = {
            "total": total,
            "per_1000": round(total / pop * 1000, 1) if pop > 0 else 0,
            "pedestrian": pedestrian,
            "cyclist": cyclist,
            "motor_vehicle": mv,
            "by_hour": by_hour,
            "by_year": dict(sorted(by_year.items())),
        }

    return result


def aggregate_violations(df):
    """Aggregate violations data by neighborhood."""
    print("Aggregating violations data...")
    result = {}

    for hood, group in df.groupby("neighborhood"):
        pop = POPULATION.get(hood, 0)
        total = len(group)

        # Top violation descriptions
        top_desc = (
            group["violation_desc"]
            .value_counts()
            .head(5)
            .reset_index()
            .rename(columns={"violation_desc": "description", "count": "count"})
        )
        top_violations = [{"description": r["description"], "count": int(r["count"])} for _, r in top_desc.iterrows()]

        result[hood] = {
            "total": total,
            "per_1000": round(total / pop * 1000, 1) if pop > 0 else 0,
            "top_violations": top_violations,
        }

    return result


# ---------------------------------------------------------------------------
# Safety score computation
# ---------------------------------------------------------------------------

def compute_safety_scores(agg_311, agg_crime, agg_crashes, agg_violations):
    """
    Compute 0-100 safety scores using percentile rank.
    Higher score = safer. Lower per-capita rate = higher score.
    """
    print("Computing safety scores...")

    # Collect all neighborhoods present in any dataset
    all_hoods = sorted(POPULATION.keys())
    n = len(all_hoods)

    raw = {}
    for hood in all_hoods:
        raw[hood] = {
            "crime_rate": agg_crime.get(hood, {}).get("per_1000", 0),
            "complaints_rate": agg_311.get(hood, {}).get("per_1000", 0),
            "crashes_rate": agg_crashes.get(hood, {}).get("per_1000", 0),
            "violations_rate": agg_violations.get(hood, {}).get("per_1000", 0),
        }

    # Percentile rank each dimension (inverted: lower rate = higher score = safer)
    dimensions = ["crime_rate", "complaints_rate", "crashes_rate", "violations_rate"]
    score_names = ["crime_score", "complaints_score", "crashes_score", "violations_score"]

    scores = {hood: {} for hood in all_hoods}

    for dim, score_name in zip(dimensions, score_names):
        values = [raw[h][dim] for h in all_hoods]
        # Rank by rate ascending (lowest rate = rank 1 = safest)
        indexed = sorted(range(n), key=lambda i: values[i])
        rank_of = [0] * n
        for rank, idx in enumerate(indexed):
            rank_of[idx] = rank
        # Convert rank to 0-100: rank 0 (lowest rate) = 100, rank n-1 (highest) = 0
        for i, hood in enumerate(all_hoods):
            if n > 1:
                score = (1 - rank_of[i] / (n - 1)) * 100
            else:
                score = 50
            scores[hood][score_name] = round(score, 1)

    # Weighted overall score
    for hood in all_hoods:
        overall = (
            WEIGHTS["crime"] * scores[hood]["crime_score"]
            + WEIGHTS["complaints"] * scores[hood]["complaints_score"]
            + WEIGHTS["crashes"] * scores[hood]["crashes_score"]
            + WEIGHTS["violations"] * scores[hood]["violations_score"]
        )
        scores[hood]["overall_score"] = round(overall, 1)

    # Add metadata
    for hood in all_hoods:
        scores[hood]["population"] = POPULATION[hood]
        scores[hood]["small_pop_warning"] = POPULATION.get(hood, 0) < SMALL_POP_THRESHOLD
        scores[hood]["crime_count"] = agg_crime.get(hood, {}).get("total", 0)
        scores[hood]["crime_per_1000"] = agg_crime.get(hood, {}).get("per_1000", 0)
        scores[hood]["complaint_count"] = agg_311.get(hood, {}).get("total", 0)
        scores[hood]["complaints_per_1000"] = agg_311.get(hood, {}).get("per_1000", 0)
        scores[hood]["violations_count"] = agg_violations.get(hood, {}).get("total", 0)
        scores[hood]["violations_per_1000"] = agg_violations.get(hood, {}).get("per_1000", 0)
        scores[hood]["crashes_count"] = agg_crashes.get(hood, {}).get("total", 0)
        scores[hood]["crashes_per_1000"] = agg_crashes.get(hood, {}).get("per_1000", 0)

    # Rank by overall score (1 = safest = highest score)
    ranked = sorted(all_hoods, key=lambda h: scores[h]["overall_score"], reverse=True)
    for i, hood in enumerate(ranked):
        scores[hood]["rank"] = i + 1

    # Per-dimension ranks (1 = safest = lowest rate)
    for rate_key, rank_key in [
        ("crime_per_1000", "crime_rank"),
        ("complaints_per_1000", "complaints_rank"),
        ("crashes_per_1000", "crashes_rank"),
        ("violations_per_1000", "violations_rank"),
    ]:
        sorted_hoods = sorted(all_hoods, key=lambda h: scores[h][rate_key])
        for i, hood in enumerate(sorted_hoods):
            scores[hood][rank_key] = i + 1

    # Store total neighborhood count for frontend display
    for hood in all_hoods:
        scores[hood]["total_neighborhoods"] = n

    return scores


# ---------------------------------------------------------------------------
# Fun fact generation
# ---------------------------------------------------------------------------

def generate_fun_facts(scores, agg_311, agg_crime, agg_crashes, agg_violations):
    """Generate a fun_fact string for each neighborhood based on its most extreme stat."""
    print("Generating fun facts...")
    all_hoods = sorted(scores.keys())
    n = len(all_hoods)
    total_pop = sum(POPULATION[h] for h in all_hoods)

    # City-wide 311 category totals (from top_topics across all neighborhoods)
    city_cat_counts = {}
    for hood in all_hoods:
        for item in agg_311.get(hood, {}).get("top_topics", []):
            city_cat_counts[item["topic"]] = city_cat_counts.get(item["topic"], 0) + item["count"]
    city_cat_per_cap = {cat: cnt / total_pop for cat, cnt in city_cat_counts.items()}

    # City-wide crash mode per-capita rates
    city_ped = sum(agg_crashes.get(h, {}).get("pedestrian", 0) for h in all_hoods)
    city_cyc = sum(agg_crashes.get(h, {}).get("cyclist", 0) for h in all_hoods)
    city_ped_per_cap = city_ped / total_pop if total_pop else 0
    city_cyc_per_cap = city_cyc / total_pop if total_pop else 0

    for hood in all_hoods:
        candidates = []  # list of (impressiveness, fact_string)
        s = scores[hood]
        pop = POPULATION[hood]

        # --- Rule 1: #1 or last in overall rank ---
        if s["rank"] == 1:
            candidates.append((n + 1, f"Ranked the safest neighborhood in Boston overall"))
        elif s["rank"] == n:
            candidates.append((n + 1, f"Ranked last in overall safety out of {n} neighborhoods"))

        # --- Rule 2: #1 or last in any dimension rank ---
        rank_dims = [
            ("crime_rank", "crime rate", "Lowest", "Highest"),
            ("complaints_rank", "311 complaint rate", "Lowest", "Highest"),
            ("crashes_rank", "crash rate", "Lowest", "Highest"),
            ("violations_rank", "violation rate", "Fewest", "Most"),
        ]
        for rank_key, label, low_word, high_word in rank_dims:
            rank = s.get(rank_key, 0)
            rate_key = rank_key.replace("_rank", "_per_1000")
            if rank_key == "complaints_rank":
                rate_key = "complaints_per_1000"
            rate = s.get(rate_key, 0)
            if rank == 1:
                candidates.append((n, f"{low_word} {label} in Boston — {rate} per 1,000 residents"))
            elif rank == n:
                candidates.append((n, f"{high_word} {label} across all Boston neighborhoods"))

        # --- Rule 3: 311 category wildly above city average (>2x) ---
        if pop > 0:
            for item in agg_311.get(hood, {}).get("top_topics", []):
                cat = item["topic"]
                if item["count"] < 10:
                    continue
                if cat in city_cat_per_cap and city_cat_per_cap[cat] > 0:
                    hood_rate = item["count"] / pop
                    multiplier = hood_rate / city_cat_per_cap[cat]
                    if multiplier > 2.0:
                        candidates.append((
                            multiplier,
                            f"{cat} complaints run {multiplier:.1f}x the city average",
                        ))

        # --- Rule 4: Peak crime hour between midnight and 4 AM ---
        by_hour = agg_crime.get(hood, {}).get("by_hour", [0] * 24)
        total_hourly = sum(by_hour)
        if total_hourly > 0:
            peak_hour = max(range(24), key=lambda h: by_hour[h])
            if 0 <= peak_hour <= 4:
                late_share = sum(by_hour[0:5]) / total_hourly
                multiplier = late_share / (5 / 24)
                if multiplier > 1.3:
                    hr_label = f"{peak_hour} AM" if peak_hour > 0 else "midnight"
                    candidates.append((
                        multiplier + 2,
                        f"Peak crime hour is {hr_label} — late-night incidents run {multiplier:.1f}x the expected rate",
                    ))

        # --- Rule 5: Year-over-year crime trend (>10% change) ---
        by_year = agg_crime.get(hood, {}).get("by_year", {})
        years_sorted = sorted(by_year.keys())
        if len(years_sorted) >= 3:
            first_year, last_year = years_sorted[0], years_sorted[-2]
        elif len(years_sorted) == 2:
            first_year, last_year = years_sorted[0], years_sorted[1]
        else:
            first_year, last_year = None, None
        if first_year and last_year and by_year.get(first_year, 0) > 0:
            pct = (by_year[last_year] - by_year[first_year]) / by_year[first_year] * 100
            if pct < -10:
                candidates.append((abs(pct) / 10, f"Crime down {abs(pct):.0f}% from {first_year} to {last_year}"))
            elif pct > 10:
                candidates.append((pct / 10, f"Crime up {pct:.0f}% from {first_year} to {last_year}"))

        # --- Rule 6: Cyclist/pedestrian crash outlier (>2x city avg) ---
        hood_crashes = agg_crashes.get(hood, {})
        if pop > 0:
            for mode, count_key, city_avg in [
                ("Pedestrian", "pedestrian", city_ped_per_cap),
                ("Cyclist", "cyclist", city_cyc_per_cap),
            ]:
                count = hood_crashes.get(count_key, 0)
                if count < 10:
                    continue
                if city_avg > 0:
                    mult = (count / pop) / city_avg
                    if mult > 2.0:
                        candidates.append((mult, f"{mode} crashes run {mult:.1f}x the city average per capita"))

        # Fallback: use the neighborhood's best-ranked dimension
        if not candidates:
            best_rank_key, best_rank_val = None, n + 1
            dim_labels = {
                "crime_rank": ("crime", "crime_per_1000"),
                "complaints_rank": ("311 complaints", "complaints_per_1000"),
                "crashes_rank": ("crash rate", "crashes_per_1000"),
                "violations_rank": ("violation rate", "violations_per_1000"),
            }
            for rk in dim_labels:
                if s.get(rk, n) < best_rank_val:
                    best_rank_val = s[rk]
                    best_rank_key = rk
            if best_rank_key:
                label, rate_key = dim_labels[best_rank_key]
                candidates.append((
                    0,
                    f"Ranks #{best_rank_val} of {n} for lowest {label} — {s.get(rate_key, 0)} per 1,000 residents",
                ))

        # Pick the single best fact
        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            scores[hood]["fun_fact"] = candidates[0][1]
            print(f"  {hood}: {candidates[0][1]}")

    count = sum(1 for h in all_hoods if "fun_fact" in scores[h])
    print(f"  Generated fun facts for {count}/{n} neighborhoods")


# ---------------------------------------------------------------------------
# GeoJSON enrichment and export
# ---------------------------------------------------------------------------

def enrich_geojson(raw_geojson, scores):
    """Embed safety scores into GeoJSON feature properties."""
    print("Enriching GeoJSON with scores...")
    for feature in raw_geojson["features"]:
        name = feature["properties"].get("neighborhood", "")
        canonical = normalize_neighborhood(name)
        if canonical and canonical in scores:
            feature["properties"].update(scores[canonical])
            feature["properties"]["canonical_name"] = canonical
        else:
            # Remove features we can't score
            feature["properties"]["canonical_name"] = None

    # Keep only features with scores
    raw_geojson["features"] = [
        f for f in raw_geojson["features"]
        if f["properties"].get("canonical_name") is not None
    ]

    # Simplify coordinates to reduce file size
    def round_coords(coords):
        if isinstance(coords[0], (int, float)):
            return [round(c, 5) for c in coords]
        return [round_coords(c) for c in coords]

    for feature in raw_geojson["features"]:
        geom = feature["geometry"]
        geom["coordinates"] = round_coords(geom["coordinates"])

    return raw_geojson


def export_json(data, filename):
    """Write JSON file with compact formatting."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  Exported {filename}: {size_kb:.1f} KB")


def export_json_to(data, filename, subdir):
    """Write JSON file to a subdirectory with compact formatting."""
    dirpath = os.path.join(DATA_DIR, subdir)
    os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  Exported {subdir}/{filename}: {size_kb:.1f} KB")


# ---------------------------------------------------------------------------
# Cambridge pipeline
# ---------------------------------------------------------------------------

def run_cambridge():
    """Run the full Cambridge data pipeline."""
    os.makedirs(CAM_DATA_DIR, exist_ok=True)

    with _cambridge_context():
        # 1. Download Cambridge neighborhood boundaries
        cam_gdf, cam_raw_geojson = download_cambridge_geojson()

        geo_names = set(cam_gdf["canonical_name"].unique())
        pop_names = set(CAM_POPULATION.keys())
        missing_in_geo = pop_names - geo_names
        extra_in_geo = geo_names - pop_names
        if missing_in_geo:
            print(f"\n  WARNING: In CAM_POPULATION but not in GeoJSON: {missing_in_geo}")
        if extra_in_geo:
            print(f"\n  WARNING: In GeoJSON but not in CAM_POPULATION: {extra_in_geo}")

        # 2. Download datasets from Socrata
        print("\n--- Downloading Cambridge datasets ---")
        df_crime = download_socrata(CAM_RESOURCES["crime"], "Cambridge Crime")
        df_311 = download_socrata(CAM_RESOURCES["311"], "Cambridge 311")
        df_crashes = download_socrata(CAM_RESOURCES["crashes"], "Cambridge Crashes")
        df_permits = download_socrata(CAM_RESOURCES["permits"], "Cambridge Permits")

        # 3. Clean
        print("\n--- Cleaning Cambridge data ---")
        df_crime = clean_cambridge_crime(df_crime)
        df_311 = clean_cambridge_311(df_311)
        df_crashes = clean_cambridge_crashes(df_crashes)
        df_permits = clean_cambridge_permits(df_permits)

        if df_311.empty and df_crime.empty:
            print("\nWARNING: Both Cambridge 311 and crime datasets are empty. Skipping Cambridge.")
            return

        # 4. Spatial join
        print("\n--- Cambridge spatial joins ---")
        if not df_311.empty:
            df_311 = assign_neighborhoods_311(df_311, cam_gdf)
        if not df_crime.empty:
            # Cambridge crime has a neighborhood column -- use it where valid,
            # spatial join only for records missing it (same pattern as 311)
            if "neighborhood" in df_crime.columns:
                has_hood = df_crime["neighborhood"].notna() & df_crime["neighborhood"].isin(CAM_POPULATION.keys())
                df_with = df_crime[has_hood].copy()
                df_without = df_crime[~has_hood].copy()
                if not df_without.empty and "lat" in df_without.columns:
                    valid_coords = df_without["lat"].notna() & df_without["lon"].notna()
                    df_need_join = df_without[valid_coords].copy()
                    if not df_need_join.empty:
                        df_joined = spatial_join(df_need_join, cam_gdf, "lat", "lon")
                        df_without = pd.concat([df_joined, df_without[~valid_coords]], ignore_index=True)
                df_crime = pd.concat([df_with, df_without], ignore_index=True)
                df_crime = df_crime[df_crime["neighborhood"].notna() & df_crime["neighborhood"].isin(CAM_POPULATION.keys())]
                print(f"  Crime after neighborhood assignment: {len(df_crime)} records")
            else:
                df_crime = spatial_join(df_crime, cam_gdf, "lat", "lon")
        if not df_permits.empty:
            df_permits = spatial_join(df_permits, cam_gdf, "latitude", "longitude")
        if not df_crashes.empty:
            df_crashes = spatial_join(df_crashes, cam_gdf, "lat", "long")

        # Filter to canonical Cambridge neighborhoods
        for name, dfx in [("crime", df_crime), ("permits", df_permits), ("crashes", df_crashes)]:
            if dfx.empty:
                continue
            valid = dfx["neighborhood"].isin(CAM_POPULATION.keys())
            invalid_hoods = dfx[~valid]["neighborhood"].unique()
            if len(invalid_hoods) > 0:
                print(f"  {name}: dropping {(~valid).sum()} records from unknown neighborhoods: {list(invalid_hoods)[:5]}")

        if not df_crime.empty:
            df_crime = df_crime[df_crime["neighborhood"].isin(CAM_POPULATION.keys())].copy()
        if not df_permits.empty:
            df_permits = df_permits[df_permits["neighborhood"].isin(CAM_POPULATION.keys())].copy()
        if not df_crashes.empty:
            df_crashes = df_crashes[df_crashes["neighborhood"].isin(CAM_POPULATION.keys())].copy()

        # 5. Aggregate
        print("\n--- Aggregating Cambridge data ---")
        agg_311 = aggregate_311(df_311) if not df_311.empty else {}
        agg_crime = aggregate_crime(df_crime) if not df_crime.empty else {}
        agg_crashes = aggregate_crashes(df_crashes) if not df_crashes.empty else {}
        agg_violations = aggregate_violations(df_permits) if not df_permits.empty else {}

        # 6. Compute scores
        print("\n--- Computing Cambridge scores ---")
        scores = compute_safety_scores(agg_311, agg_crime, agg_crashes, agg_violations)

        # 7. Generate fun facts
        print("\n--- Generating Cambridge fun facts ---")
        generate_fun_facts(scores, agg_311, agg_crime, agg_crashes, agg_violations)

        # Post-process: replace "Boston" with "Cambridge" in fun facts
        for hood in scores:
            if "fun_fact" in scores[hood]:
                scores[hood]["fun_fact"] = scores[hood]["fun_fact"].replace("Boston", "Cambridge")

        # Print Cambridge rankings
        print("\n--- Cambridge Safety Score Rankings ---")
        ranked = sorted(scores.items(), key=lambda x: x[1]["rank"])
        for hood, s in ranked:
            flag = " *" if s["small_pop_warning"] else ""
            print(f"  #{s['rank']:2d} {hood:30s} Overall: {s['overall_score']:5.1f}  "
                  f"Crime: {s['crime_score']:5.1f}  311: {s['complaints_score']:5.1f}  "
                  f"Crashes: {s['crashes_score']:5.1f}  Violations: {s['violations_score']:5.1f}{flag}")

        # 8. Enrich GeoJSON
        enriched = enrich_geojson(cam_raw_geojson, scores)

    # 9. Export to data/cambridge/
    print("\n--- Exporting Cambridge data ---")
    export_json_to(scores, "safety_scores.json", "cambridge")
    export_json_to(agg_311, "311_summary.json", "cambridge")
    export_json_to(agg_crime, "crime_summary.json", "cambridge")
    export_json_to(agg_crashes, "crashes_summary.json", "cambridge")
    export_json_to(agg_violations, "violations_summary.json", "cambridge")
    export_json_to(enriched, "neighborhoods.geojson", "cambridge")

    # Verify files
    for fname in ["safety_scores.json", "311_summary.json", "crime_summary.json",
                   "crashes_summary.json", "violations_summary.json", "neighborhoods.geojson"]:
        fpath = os.path.join(CAM_DATA_DIR, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            print(f"  OK: cambridge/{fname} ({os.path.getsize(fpath) / 1024:.1f} KB)")
        else:
            print(f"  MISSING or EMPTY: cambridge/{fname}")

    print("\nCambridge pipeline complete!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Download neighborhood boundaries
    neighborhoods_gdf, raw_geojson = download_geojson()

    # Log canonical names vs population keys
    geo_names = set(neighborhoods_gdf["canonical_name"].unique())
    pop_names = set(POPULATION.keys())
    missing_in_geo = pop_names - geo_names
    extra_in_geo = geo_names - pop_names
    if missing_in_geo:
        print(f"\n  WARNING: In POPULATION but not in GeoJSON: {missing_in_geo}")
    if extra_in_geo:
        print(f"\n  WARNING: In GeoJSON but not in POPULATION: {extra_in_geo}")

    # 2. Download datasets
    print("\n--- Downloading datasets ---")
    # 311 has multiple yearly resources
    dfs_311 = []
    for i, rid in enumerate(RESOURCES["311"]):
        label = f"311 Service Requests ({2023 + i})"
        try:
            dfs_311.append(download_datastore(rid, label))
        except Exception as e:
            print(f"  Datastore API failed for {label}: {e}")
            try:
                dfs_311.append(download_csv_direct(rid, label))
            except Exception as e2:
                print(f"  CSV download also failed for {label}: {e2}")
    df_311 = pd.concat(dfs_311, ignore_index=True) if dfs_311 else pd.DataFrame()

    try:
        df_crime = download_datastore(RESOURCES["crime"], "Crime Incidents")
    except Exception as e:
        print(f"  Datastore API failed for crime: {e}")
        print("  Trying direct CSV download...")
        df_crime = download_csv_direct(RESOURCES["crime"], "Crime Incidents")

    try:
        df_violations = download_datastore(RESOURCES["violations"], "Violations")
    except Exception as e:
        print(f"  Datastore API failed for violations: {e}")
        df_violations = download_csv_direct(RESOURCES["violations"], "Violations")

    try:
        df_crashes = download_datastore(RESOURCES["crashes"], "Crash Records")
    except Exception as e:
        print(f"  Datastore API failed for crashes: {e}")
        df_crashes = download_csv_direct(RESOURCES["crashes"], "Crash Records")

    # 3. Clean
    print("\n--- Cleaning data ---")
    df_311 = clean_311(df_311)
    df_crime = clean_crime(df_crime)
    df_violations = clean_violations(df_violations)
    df_crashes = clean_crashes(df_crashes)

    if df_311.empty or df_crime.empty:
        print("\nERROR: Critical dataset is empty after cleaning. Aborting.")
        sys.exit(1)

    # 4. Spatial join
    print("\n--- Spatial joins ---")
    df_311 = assign_neighborhoods_311(df_311, neighborhoods_gdf)
    df_crime = spatial_join(df_crime, neighborhoods_gdf, "lat", "lon")
    if not df_violations.empty:
        df_violations = spatial_join(df_violations, neighborhoods_gdf, "latitude", "longitude")
    if not df_crashes.empty:
        df_crashes = spatial_join(df_crashes, neighborhoods_gdf, "lat", "long")

    # Filter to canonical neighborhoods
    for name, dfx in [("crime", df_crime), ("violations", df_violations), ("crashes", df_crashes)]:
        valid = dfx["neighborhood"].isin(POPULATION.keys())
        invalid_hoods = dfx[~valid]["neighborhood"].unique()
        if len(invalid_hoods) > 0:
            print(f"  {name}: dropping {(~valid).sum()} records from unknown neighborhoods: {invalid_hoods[:5]}")

    df_crime = df_crime[df_crime["neighborhood"].isin(POPULATION.keys())].copy()
    if not df_violations.empty:
        df_violations = df_violations[df_violations["neighborhood"].isin(POPULATION.keys())].copy()
    if not df_crashes.empty:
        df_crashes = df_crashes[df_crashes["neighborhood"].isin(POPULATION.keys())].copy()

    # 5. Aggregate
    print("\n--- Aggregating ---")
    agg_311 = aggregate_311(df_311)
    agg_crime = aggregate_crime(df_crime)
    agg_crashes = aggregate_crashes(df_crashes) if not df_crashes.empty else {}
    agg_violations = aggregate_violations(df_violations) if not df_violations.empty else {}

    # 6. Compute scores
    print("\n--- Computing scores ---")
    scores = compute_safety_scores(agg_311, agg_crime, agg_crashes, agg_violations)

    # 6b. Generate fun facts
    print("\n--- Generating fun facts ---")
    generate_fun_facts(scores, agg_311, agg_crime, agg_crashes, agg_violations)

    # Print summary
    print("\n--- Safety Score Rankings ---")
    ranked = sorted(scores.items(), key=lambda x: x[1]["rank"])
    for hood, s in ranked:
        flag = " *" if s["small_pop_warning"] else ""
        print(f"  #{s['rank']:2d} {hood:30s} Overall: {s['overall_score']:5.1f}  "
              f"Crime: {s['crime_score']:5.1f}  311: {s['complaints_score']:5.1f}  "
              f"Crashes: {s['crashes_score']:5.1f}  Violations: {s['violations_score']:5.1f}{flag}")

    # 7. Enrich GeoJSON
    enriched = enrich_geojson(raw_geojson, scores)

    # 8. Export
    print("\n--- Exporting ---")
    export_json(scores, "safety_scores.json")
    export_json(agg_311, "311_summary.json")
    export_json(agg_crime, "crime_summary.json")
    export_json(agg_crashes, "crashes_summary.json")
    export_json(agg_violations, "violations_summary.json")
    export_json(enriched, "neighborhoods.geojson")

    # Total size
    total_kb = sum(
        os.path.getsize(os.path.join(DATA_DIR, f)) / 1024
        for f in os.listdir(DATA_DIR)
        if f.endswith((".json", ".geojson"))
    )
    print(f"\n  Total data size: {total_kb:.1f} KB")
    print("\nDone with Boston pipeline!")

    # --- Cambridge pipeline ---
    print("\n" + "=" * 60)
    print("=== Running Cambridge pipeline ===")
    print("=" * 60)
    try:
        run_cambridge()
    except Exception as e:
        print(f"\nCambridge pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        print("Boston data is unaffected.")


if __name__ == "__main__":
    main()
