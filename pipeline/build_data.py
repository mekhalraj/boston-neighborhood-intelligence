"""
Boston Neighborhood Safety Map -- Data Pipeline
Downloads data from Analyze Boston, processes it, and exports JSON files.
"""

import json
import os
import sys
import time
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
    "311": "254adca6-64ab-4c5c-9fc0-a6da622be185",
    "crime": "b973d8cb-eeb2-4e7e-99da-c92938efc9c0",
    "violations": "800a2663-1d6a-46e7-9356-bedb70f5332c",
    "crashes": "e4bfe397-6bfc-49c5-9367-c879fac7401d",
}

NEIGHBORHOODS_GEOJSON_URL = (
    "https://data.boston.gov/dataset/bf1a7b50-4c72-4637-b0fa-11d632e3aff1/"
    "resource/e5849875-a6f6-4c9c-9d8a-5048b0fbd03e/download/"
    "boston_neighborhood_boundaries.geojson"
)

# 2020 Census / ACS approximate populations
POPULATION = {
    "Allston": 29400,
    "Back Bay": 26200,
    "Bay Village": 1500,
    "Beacon Hill": 10200,
    "Brighton": 45000,
    "Charlestown": 18900,
    "Chinatown": 8900,
    "Dorchester": 128500,
    "Downtown": 13800,
    "East Boston": 47800,
    "Fenway": 39500,
    "Hyde Park": 37400,
    "Jamaica Plain": 41200,
    "Mattapan": 27300,
    "Mission Hill": 18100,
    "North End": 10600,
    "Roslindale": 33600,
    "Roxbury": 55800,
    "South Boston": 36300,
    "South Boston Waterfront": 5300,
    "South End": 34100,
    "West End": 4900,
    "West Roxbury": 34500,
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
NAME_ALIASES = {
    # Common variants
    "South Boston Waterfront": "South Boston Waterfront",
    "Seaport": "South Boston Waterfront",
    "Downtown / Financial District": "Downtown",
    "Financial District": "Downtown",
    "Longwood Medical and Academic Area": "Fenway",
    "Longwood Medical Area": "Fenway",
    "Longwood": "Fenway",
    "Leather District": "Downtown",
    "Harbor Islands": None,  # exclude
    "West End": "West End",
}


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
    print(f"  Neighborhoods found: {sorted(gdf['name'].unique())}")
    gdf["canonical_name"] = gdf["name"].apply(normalize_neighborhood)
    gdf = gdf[gdf["canonical_name"].notna()].copy()
    print(f"  Canonical neighborhoods: {sorted(gdf['canonical_name'].unique())}")
    return gdf, r.json()


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
        pop = POPULATION.get(hood, 1)
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
            "per_1000": round(total / pop * 1000, 1),
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
        pop = POPULATION.get(hood, 1)
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
            "per_1000": round(total / pop * 1000, 1),
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
        pop = POPULATION.get(hood, 1)
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
            "per_1000": round(total / pop * 1000, 1),
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
        pop = POPULATION.get(hood, 1)
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
            "per_1000": round(total / pop * 1000, 1),
            "top_violations": top_violations,
        }

    return result


# ---------------------------------------------------------------------------
# Safety score computation
# ---------------------------------------------------------------------------

def compute_safety_scores(agg_311, agg_crime, agg_crashes, agg_violations):
    """
    Compute 0-100 safety scores using z-score normalization.
    Higher score = safer.
    """
    print("Computing safety scores...")

    # Collect all neighborhoods present in any dataset
    all_hoods = set(POPULATION.keys())

    raw = {}
    for hood in all_hoods:
        raw[hood] = {
            "crime_rate": agg_crime.get(hood, {}).get("per_1000", 0),
            "complaints_rate": agg_311.get(hood, {}).get("per_1000", 0),
            "crashes_rate": agg_crashes.get(hood, {}).get("per_1000", 0),
            "violations_rate": agg_violations.get(hood, {}).get("per_1000", 0),
        }

    # Z-score normalize each dimension (inverted: lower rate = higher z = safer)
    dimensions = ["crime_rate", "complaints_rate", "crashes_rate", "violations_rate"]
    score_names = ["crime_score", "complaints_score", "crashes_score", "violations_score"]

    scores = {hood: {} for hood in all_hoods}

    for dim, score_name in zip(dimensions, score_names):
        values = np.array([raw[h][dim] for h in all_hoods])
        mean_val = np.mean(values)
        std_val = np.std(values)

        for hood in all_hoods:
            if std_val > 0:
                z = -(raw[hood][dim] - mean_val) / std_val  # negative: lower rate = higher score
            else:
                z = 0
            # Convert z-score to 0-100 scale: z of -2 -> 0, z of +2 -> 100
            score = max(0, min(100, (z + 2) * 25))
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
        scores[hood]["small_pop_warning"] = POPULATION[hood] < SMALL_POP_THRESHOLD
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

    return scores


# ---------------------------------------------------------------------------
# GeoJSON enrichment and export
# ---------------------------------------------------------------------------

def enrich_geojson(raw_geojson, scores):
    """Embed safety scores into GeoJSON feature properties."""
    print("Enriching GeoJSON with scores...")
    for feature in raw_geojson["features"]:
        name = feature["properties"].get("name", "")
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
    try:
        df_311 = download_datastore(RESOURCES["311"], "311 Service Requests")
    except Exception as e:
        print(f"  Datastore API failed for 311: {e}")
        print("  Trying direct CSV download...")
        df_311 = download_csv_direct(RESOURCES["311"], "311 Service Requests")

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
    print("\nDone!")


if __name__ == "__main__":
    main()
