### COMMENTS OF THE CODE AND VARIABLE NAMES WERE REVIEWED AND IMPROVED BY CHATGPT-5 ###
### ORIGINALLY STRUCTURED AS A NOTEBOOK, AGGREGATED TO A SINGLE SCRIPT ###

from __future__ import annotations

# Core typing and utilities
from typing import Iterable, Dict, Optional, Tuple, List, Set
import os
import re

# Data and geo
import pandas as pd
import geopandas as gpd
import fiona
from IPython.display import display

# ==========================
# Paths (edit if needed)
# ==========================
TSV_PATH = "data/gadm1_nuts3_counties-gadm1_nuts3_counties - FB Social Connectedness Index - October 2021.tsv"
MAP_PATH = "data/gadm1_nuts3_counties_levels.csv"

NUTS_GEOJSON_PATH = "data/NUTS_RG_60M_2016_4326_LEVL_3.geojson"
US_COUNTIES_PATH = "data/us-county-boundaries.geojson"
GADM_GPKG_PATH = "data/gadm_410.gpkg"  

SCI_COLS = ["user_loc", "fr_loc", "scaled_sci"]
DTYPE_MAP = {"user_loc": "string", "fr_loc": "string", "scaled_sci": "float64"}


# ==========================
# Small helpers
# ==========================

def _normalize_location_code(s: pd.Series) -> pd.Series:
    """Normalize codes to uppercase strings with no surrounding spaces."""
    return s.astype("string").str.strip().str.upper()


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], where: str = "") -> None:
    """Check that a DataFrame contains the required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        loc = f" in {where}" if where else ""
        raise ValueError(f"Missing columns{loc}: {missing}")


def _preview(df: pd.DataFrame, name: str = "DataFrame", n: int = 5) -> None:
    """Show a quick preview: head, dtypes, and row count."""
    print(f"== {name} – head ==")
    display(df.head(n))
    print("\nSchema:")
    print(df.dtypes)
    print(f"\nRows: {len(df):,}")


# ==========================
# Load SCI and mapping files
# ==========================

def load_sci_tsv(
    path: str,
    sci_cols: Iterable[str] = ("user_loc", "fr_loc", "scaled_sci"),
    dtype_map: Optional[Dict[str, str]] = None,
    low_memory: bool = False
) -> pd.DataFrame:
    """Load the FB SCI TSV with selected columns and dtypes."""
    if dtype_map is None:
        dtype_map = {"user_loc": "string", "fr_loc": "string", "scaled_sci": "float64"}

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=list(sci_cols),
        dtype=dtype_map,
        low_memory=low_memory
    )

    if "user_loc" in df.columns:
        df["user_loc"] = _normalize_location_code(df["user_loc"])
    if "fr_loc" in df.columns:
        df["fr_loc"] = _normalize_location_code(df["fr_loc"])

    _ensure_columns(df, sci_cols, where="SCI TSV")
    return df


def load_levels_mapping(
    path: str,
    usecols: Iterable[str] = ("key", "level"),
    rename_map: Dict[str, str] = {"key": "location_code", "level": "level_type"},
) -> pd.DataFrame:
    """Load the level mapping (location_code, level_type)."""
    df = pd.read_csv(path, usecols=list(usecols), dtype="string").rename(columns=rename_map)
    _ensure_columns(df, ["location_code", "level_type"], where="Levels mapping")
    df["location_code"] = _normalize_location_code(df["location_code"])
    return df


def quick_read_inputs(
    tsv_path: str,
    map_path: str,
    sci_cols: Iterable[str] = ("user_loc", "fr_loc", "scaled_sci"),
    dtype_map: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both SCI and level mapping and preview them."""
    df_sci = load_sci_tsv(tsv_path, sci_cols=sci_cols, dtype_map=dtype_map)
    _preview(df_sci, name="SCI (TSV)")
    df_map = load_levels_mapping(map_path)
    _preview(df_map, name="Levels mapping")
    return df_sci, df_map


# Load inputs
df_sci, df_map = quick_read_inputs(TSV_PATH, MAP_PATH, sci_cols=SCI_COLS, dtype_map=DTYPE_MAP)

print("\n== level_type value_counts ==")
if "level_type" in df_map:
    display(df_map['level_type'].value_counts())


# ==========================
# Coverage metrics
# ==========================

def get_unique_location_codes(df_sci: pd.DataFrame) -> pd.Series:
    """Return unique FB location codes (user_loc ∪ fr_loc)."""
    _ensure_columns(df_sci, ["user_loc", "fr_loc"], where="SCI")
    u = _normalize_location_code(df_sci["user_loc"])
    v = _normalize_location_code(df_sci["fr_loc"])
    return pd.Index(u).append(pd.Index(v)).astype("string").unique()


def get_mapped_codes(df_map: pd.DataFrame) -> pd.Series:
    """Return unique mapped codes from the mapping table."""
    _ensure_columns(df_map, ["location_code"], where="Levels mapping")
    return _normalize_location_code(df_map["location_code"]).dropna().unique()


def compute_node_coverage(df_sci: pd.DataFrame, df_map: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute how many nodes in SCI are mapped and list the missing ones."""
    sci_codes = pd.Series(get_unique_location_codes(df_sci), name="location_code")
    map_codes = pd.Series(get_mapped_codes(df_map), name="location_code")

    total = sci_codes.size
    mapped_mask = sci_codes.isin(set(map_codes))
    mapped = int(mapped_mask.sum())
    unmapped = (
        sci_codes[~mapped_mask].to_frame().drop_duplicates()
        .sort_values("location_code").reset_index(drop=True)
    )

    summary = {
        "total_unique_codes": int(total),
        "mapped_unique_codes": int(mapped),
        "unmapped_unique_codes": int(total - mapped),
        "node_coverage_pct": (mapped / total * 100.0) if total else 0.0,
    }
    return unmapped, summary


def compute_edge_coverage(df_sci: pd.DataFrame, df_map: pd.DataFrame) -> Dict[str, float]:
    """Compute the share of edges where both endpoints are mapped."""
    _ensure_columns(df_sci, ["user_loc", "fr_loc"], where="SCI")
    mapped_set = set(get_mapped_codes(df_map))

    u = _normalize_location_code(df_sci["user_loc"])
    v = _normalize_location_code(df_sci["fr_loc"])

    mask = u.isin(mapped_set) & v.isin(mapped_set)
    total_rows = int(len(df_sci))
    valid_rows = int(mask.sum())

    return {
        "total_rows": total_rows,
        "valid_rows_both_mapped": valid_rows,
        "edge_coverage_pct": (valid_rows / total_rows * 100.0) if total_rows else 0.0,
    }


def compute_country_coverage(
    df_sci: pd.DataFrame,
    df_map: pd.DataFrame,
    iso_col: str = "country_ISO3",
    keep_only_intra_country: bool = True
) -> Optional[pd.DataFrame]:
    """
    Aggregate SCI edges by country using the mapping table.
    If iso_col is missing, return None.
    """
    if iso_col not in df_map.columns:
        print(f"[WARN] Column '{iso_col}' not found in mapping: country coverage not computed.")
        return None

    _ensure_columns(df_sci, ["user_loc", "fr_loc"], where="SCI")

    df_map_local = df_map[["location_code", iso_col]].copy()
    df_map_local["location_code"] = _normalize_location_code(df_map_local["location_code"])
    code2iso = (
        df_map_local.dropna().drop_duplicates("location_code")
        .set_index("location_code")[iso_col]
    )

    sci = df_sci[["user_loc", "fr_loc"]].copy()
    sci["user_loc"] = _normalize_location_code(sci["user_loc"])
    sci["fr_loc"] = _normalize_location_code(sci["fr_loc"])

    sci["iso_from"] = sci["user_loc"].map(code2iso)
    sci["iso_to"] = sci["fr_loc"].map(code2iso)

    sci_valid = sci.dropna(subset=["iso_from", "iso_to"]).copy()
    if keep_only_intra_country:
        sci_valid = sci_valid[sci_valid["iso_from"] == sci_valid["iso_to"]].copy()

    edges_by_country = (
        sci_valid.groupby("iso_from", as_index=False)
        .agg(edges=("user_loc", "size")).rename(columns={"iso_from": iso_col})
    )

    nodes_from = sci_valid[["iso_from", "user_loc"]].rename(columns={"iso_from": iso_col, "user_loc": "loc"})
    nodes_to = sci_valid[["iso_to", "fr_loc"]].rename(columns={"iso_to": iso_col, "fr_loc": "loc"})
    nodes_all = pd.concat([nodes_from, nodes_to], ignore_index=True).drop_duplicates()

    nodes_by_country = nodes_all.groupby(iso_col, as_index=False).agg(nodes=("loc", "nunique"))

    country_cov = edges_by_country.merge(nodes_by_country, on=iso_col, how="outer").fillna(0)
    country_cov["edges"] = country_cov["edges"].astype(int)
    country_cov["nodes"] = country_cov["nodes"].astype(int)

    total_edges = int(country_cov["edges"].sum()) if len(country_cov) else 0
    country_cov["edges_pct"] = (100.0 * country_cov["edges"] / total_edges) if total_edges else 0.0

    return country_cov.sort_values(["edges", "nodes"], ascending=False).reset_index(drop=True)


def coverage_report(df_sci: pd.DataFrame, df_map: pd.DataFrame, top_n: int = 10) -> None:
    """Print a compact coverage report for nodes, edges, and countries."""
    unmapped_nodes, node_summary = compute_node_coverage(df_sci, df_map)
    print("== NODE COVERAGE ==")
    for k, v in node_summary.items():
        print(f"- {k}: {v}")
    if len(unmapped_nodes) > 0:
        print(f"\nExamples of UNMAPPED codes ({min(10, len(unmapped_nodes))}):")
        display(unmapped_nodes.head(10))

    edge_summary = compute_edge_coverage(df_sci, df_map)
    print("\n== EDGE COVERAGE ==")
    for k, v in edge_summary.items():
        print(f"- {k}: {v}")

    country_cov = compute_country_coverage(df_sci, df_map)
    if country_cov is not None and len(country_cov):
        print("\n== COVERAGE BY COUNTRY (head) ==")
        display(country_cov.head(top_n))
    else:
        print("\n== COVERAGE BY COUNTRY ==")
        print("Mapping does not contain 'country_ISO3' → add it to df_map if needed.")


# Run coverage report
coverage_report(df_sci, df_map)


# ==========================
# Geo helpers and loaders
# ==========================

def _representative_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert to EPSG:4326 and set representative points and lat/lon."""
    gdf = gdf.to_crs(4326)
    gdf["geometry"] = gdf["geometry"].representative_point()
    gdf["latitude"] = gdf.geometry.y
    gdf["longitude"] = gdf.geometry.x
    return gdf


def load_nuts3_points(nuts_geojson_path: str, code_col: Optional[str] = None) -> pd.DataFrame:
    """Load NUTS3 polygons and return a centroid-like point table (code, name, lat, lon)."""
    gdf = gpd.read_file(nuts_geojson_path)

    # Keep only level 3 features if the column exists
    levl_col = next((c for c in ["LEVL_CODE", "LEVL", "LEVEL"] if c in gdf.columns), None)
    if levl_col is not None:
        gdf = gdf[gdf[levl_col].astype(str).isin(["3", 3])].copy()

    candidates_code = [code_col] if code_col else [c for c in ["NUTS_ID", "nuts_id", "ID", "id"] if c in gdf.columns]
    if not candidates_code:
        raise ValueError("Missing a NUTS code column (e.g., 'NUTS_ID'). Pass code_col=...")

    cc = candidates_code[0]
    name_col = next((c for c in ["NUTS_NAME", "NAME_LATN", "NAME_ENGL", "NAME"] if c in gdf.columns), None)

    gdf = gdf.rename(columns={cc: "code", (name_col if name_col else cc): "name"})
    cols = ["code", "name", "geometry"] if name_col else ["code", "geometry"]
    gdf = _representative_points(gdf[cols].dropna(subset=["code"]))

    gdf["code"] = _normalize_location_code(gdf["code"])
    if "name" not in gdf:
        gdf["name"] = pd.NA

    return gdf[["code", "name", "latitude", "longitude"]].drop_duplicates("code")


def load_gadm2_points(gadm_gpkg_path: str, layer: str = "gadm_410", code_col: str = "GID_2") -> pd.DataFrame:
    """Load GADM level-2 polygons and return point table (code, name, lat, lon)."""
    gdf = gpd.read_file(gadm_gpkg_path, layer=layer)
    if code_col not in gdf.columns:
        raise ValueError(f"Column {code_col!r} not found in layer {layer}. Found: {list(gdf.columns)}")

    name_col = "NAME_2" if "NAME_2" in gdf.columns else None

    gdf2 = gdf[~gdf[code_col].isna()].copy().rename(columns={code_col: "code"})
    if name_col:
        gdf2 = gdf2.rename(columns={name_col: "name"})
    gdf2 = gdf2.to_crs(4326)
    gdf2["geometry"] = gdf2.geometry.representative_point()
    gdf2["latitude"] = gdf2.geometry.y
    gdf2["longitude"] = gdf2.geometry.x
    gdf2["code"] = gdf2["code"].astype("string").str.strip().str.upper()
    if "name" not in gdf2:
        gdf2["name"] = pd.NA

    out = gdf2[["code", "name", "latitude", "longitude"]].drop_duplicates("code")
    print(f"[GADM2] Loaded {len(out):,} unique ADM2 units from {gadm_gpkg_path}")
    return out


def load_us_counties_points(counties_geojson_path: str, code_col: Optional[str] = None) -> pd.DataFrame:
    """Load US counties and return point table (code=GEOID 5 digits, name, lat, lon)."""
    gdf = gpd.read_file(counties_geojson_path)
    candidates = [code_col] if code_col else [c for c in ["GEOID", "geoid", "FIPS", "fips"] if c in gdf.columns]
    if not candidates:
        raise ValueError("Missing a counties code column (e.g., 'GEOID'). Pass code_col=...")

    cc = candidates[0]
    name_col = next((c for c in ["NAME", "NAMELSAD", "name"] if c in gdf.columns), None)

    gdf = gdf.rename(columns={cc: "code", (name_col if name_col else cc): "name"})
    gdf["code"] = gdf["code"].astype(str).str.zfill(5)
    gdf = _representative_points(gdf[["code", "name", "geometry"]].dropna(subset=["code"]))
    gdf["code"] = _normalize_location_code(gdf["code"])

    return gdf[["code", "name", "latitude", "longitude"]].drop_duplicates("code")


def load_gadm1_points(gadm_gpkg_path: str, layer: str = "gadm_410", code_col: str = "GID_1") -> pd.DataFrame:
    """Load GADM level-1 polygons and return point table (code, name, lat, lon)."""
    gdf = gpd.read_file(gadm_gpkg_path, layer=layer)
    if code_col not in gdf.columns:
        raise ValueError(f"Column {code_col!r} not found in {layer}. Columns: {list(gdf.columns)}")
    name_col = "NAME_1" if "NAME_1" in gdf.columns else None

    gdf1 = gdf[~gdf[code_col].isna()].copy().rename(columns={code_col: "code"})
    if name_col:
        gdf1 = gdf1.rename(columns={name_col: "name"})
    gdf1 = gdf1.to_crs(4326)
    gdf1["geometry"] = gdf1.geometry.representative_point()
    gdf1["latitude"] = gdf1.geometry.y
    gdf1["longitude"] = gdf1.geometry.x
    gdf1["code"] = gdf1["code"].astype("string").str.strip().str.upper()
    if "name" not in gdf1:
        gdf1["name"] = pd.NA

    return gdf1[["code", "name", "latitude", "longitude"]].drop_duplicates("code")


# ==========================
# Select targets from mapping
# ==========================

def select_target_codes(
    df_sci: pd.DataFrame,
    df_map: pd.DataFrame,
    level_types: Iterable[str] = ("NUTS3", "GADM2", "GADM1", "COUNTY"),
    sci_required: Iterable[str] = ("user_loc", "fr_loc"),
    map_required: Iterable[str] = ("location_code", "level_type"),
) -> pd.DataFrame:
    """
    Keep only mapping rows that:
      - belong to requested level types
      - are present in SCI (user_loc ∪ fr_loc)
    """
    _ensure_columns(df_sci, sci_required, "SCI")
    _ensure_columns(df_map, map_required, "Mapping")

    sci_codes = pd.Index(_normalize_location_code(df_sci["user_loc"])) \
        .append(pd.Index(_normalize_location_code(df_sci["fr_loc"]))) \
        .unique()

    df_map2 = df_map.copy()
    df_map2["location_code"] = _normalize_location_code(df_map2["location_code"])
    df_map2["level_type"] = df_map2["level_type"].astype("string")

    wanted = {t.upper() for t in level_types}
    df_map2 = df_map2[df_map2["level_type"].str.upper().isin(wanted)]

    target = (df_map2[df_map2["location_code"].isin(set(sci_codes))]
              .drop_duplicates(subset=["location_code", "level_type"])
              .reset_index(drop=True))

    print(f"[INFO] Selected target codes: {len(target):,} "
          f"(types: {sorted(target['level_type'].str.upper().unique())})")
    return target


# ==========================
# Normalize mapping codes for joins
# ==========================

def normalize_gadm1_mapping_code(code: Optional[str]) -> Optional[str]:
    """
    Expected input: 'BGD1', 'IND23', or already 'BGD.1_1'.
    Output (GADM v4): 'ISO3.ADM1_1' (e.g., 'BGD.1_1').
    """
    if code is None or pd.isna(code):
        return None
    s = str(code).strip().upper()
    if re.fullmatch(r"[A-Z]{3}\.\d+_1", s):
        return s
    m = re.fullmatch(r'([A-Z]{3})(\d+)', s)
    if not m:
        return None
    iso3, adm1 = m.groups()
    return f"{iso3}.{int(adm1)}_1"


def normalize_gadm2_mapping_code(code: Optional[str]) -> Optional[str]:
    """
    Convert to GADM v4 GID_2 'ISO3.A.B_1' (e.g., 'AGO.4.7_1').
    Accepts flexible forms like 'AGO4-7', 'AGO4.7', 'AGO-4-7', 'AGO4_7'.
    """
    if code is None or pd.isna(code):
        return None
    s = str(code).strip().str.upper()
    if re.fullmatch(r"[A-Z]{3}\.\d+\.\d+_1", s):
        return s
    m = re.fullmatch(r"([A-Z]{3})[\.-_ ]?(\d+)[\.-_ ]?(\d+)", s)
    if not m:
        return None
    iso3, adm1, adm2 = m.groups()
    return f"{iso3}.{int(adm1)}.{int(adm2)}_1"


def normalize_county_mapping_code(code: Optional[str]) -> Optional[str]:
    """
    Convert to 5-digit GEOID for US counties.
    Example: 'USA06091' -> '06091'.
    """
    if code is None or pd.isna(code):
        return None
    s = str(code).strip().upper()
    if s.startswith("USA"):
        s = s[3:]
    s = re.sub(r'\D', '', s)
    if len(s) == 5:
        return s
    if 1 <= len(s) <= 5:
        return s.zfill(5)
    return None


# ==========================
# Build nodes per type and assemble final list
# ==========================

def build_nodes_for_type_with_transform(
    target_codes: pd.DataFrame,
    type_name: str,
    source_df_points: pd.DataFrame,
    transform_fn=None
) -> pd.DataFrame:
    """Join target codes with geodata points and produce node rows."""
    mask = target_codes["level_type"].str.upper() == type_name.upper()
    wanted = target_codes.loc[mask, ["location_code"]].copy()
    wanted["location_code"] = wanted["location_code"].astype("string").str.strip().str.upper()

    if transform_fn is not None:
        wanted["code"] = wanted["location_code"].map(transform_fn)
    else:
        wanted["code"] = wanted["location_code"]

    wanted = wanted.dropna(subset=["code"]).drop_duplicates("code")

    pts = source_df_points.copy()
    pts["code"] = pts["code"].astype("string").str.strip().str.upper()

    out = wanted.merge(pts, on="code", how="inner")

    t = type_name.upper()
    if t == "COUNTY":
        out["nodeLabel"] = "USA" + out["code"].astype(str).str.zfill(5)
    else:
        out["nodeLabel"] = out["code"]

    out["DatasetDiOrigine"] = t
    if "name" in out.columns:
        out = out.rename(columns={"name": "nodeName"})
    else:
        out["nodeName"] = pd.NA

    cols = ["nodeLabel", "nodeName", "latitude", "longitude", "DatasetDiOrigine"]
    return out[cols].drop_duplicates("nodeLabel")


def _gid1_base_from_gid1(gid1: str) -> Optional[str]:
    """Extract 'ISO3.A' from 'ISO3.A_1'."""
    if not isinstance(gid1, str):
        return None
    s = gid1.strip().upper()
    m = re.fullmatch(r"([A-Z]{3}\.\d+)_1", s)
    return m.group(1) if m else None


def _gid1_base_from_gid2(gid2: str) -> Optional[str]:
    """Extract 'ISO3.A' from 'ISO3.A.B_1'."""
    if not isinstance(gid2, str):
        return None
    s = gid2.strip().upper()
    m = re.fullmatch(r"([A-Z]{3}\.\d+)\.\d+_1", s)
    return m.group(1) if m else None


def drop_gadm1_if_gadm2_present(gadm1_nodes: pd.DataFrame, gadm2_nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Drop GADM1 nodes when there is any GADM2 node in the same ADM1.
    This prefers finer granularity when available.
    """
    if gadm1_nodes is None or len(gadm1_nodes) == 0:
        return gadm1_nodes
    if gadm2_nodes is None or len(gadm2_nodes) == 0:
        return gadm1_nodes

    g1 = gadm1_nodes.copy()
    g2 = gadm2_nodes.copy()

    g1["_gid1base"] = g1["nodeLabel"].map(_gid1_base_from_gid1)
    g2_bases: Set[str] = set(g2["nodeLabel"].map(_gid1_base_from_gid2).dropna().unique())

    keep_mask = ~g1["_gid1base"].isin(g2_bases)
    kept = g1.loc[keep_mask, [c for c in g1.columns if c != "_gid1base"]].copy()
    dropped = len(g1) - len(kept)
    if dropped:
        print(f"[INFO] Removed {dropped:,} GADM1 nodes covered by GADM2 in the same ADM1")
    return kept


def assemble_node_list(
    nuts_nodes: Optional[pd.DataFrame] = None,
    gadm2_nodes: Optional[pd.DataFrame] = None,
    gadm1_nodes: Optional[pd.DataFrame] = None,
    county_nodes: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Merge node frames with priority:
      NUTS3 + GADM2 + (GADM1 filtered by GADM2 presence) + COUNTY.
    Assign nodeID and keep required columns.
    """
    if gadm1_nodes is not None and gadm2_nodes is not None:
        gadm1_nodes = drop_gadm1_if_gadm2_present(gadm1_nodes, gadm2_nodes)

    frames = [df for df in [nuts_nodes, gadm2_nodes, gadm1_nodes, county_nodes] if df is not None and len(df)]
    if not frames:
        raise ValueError("No nodes provided.")

    nodes = pd.concat(frames, ignore_index=True)
    nodes = nodes.drop_duplicates("nodeLabel")
    nodes = nodes.sort_values("nodeLabel").reset_index(drop=True)
    nodes.insert(0, "nodeID", range(1, len(nodes) + 1))

    return nodes[["nodeID", "nodeLabel", "nodeName", "latitude", "longitude", "DatasetDiOrigine"]]


# ==========================
# Quick layer inspection (optional)
# ==========================
layers = fiona.listlayers(GADM_GPKG_PATH)
print(f"Layers in {GADM_GPKG_PATH}:")
for i, lyr in enumerate(layers, 1):
    print(f"{i:>2}. {lyr}")

# Small previews (safe fallback if 'rows' is unsupported)
try:
    nuts_sample = gpd.read_file(NUTS_GEOJSON_PATH, rows=5)
except TypeError:
    nuts_sample = gpd.read_file(NUTS_GEOJSON_PATH)
print("\nColumns in NUTS3 GeoJSON:")
print(list(nuts_sample.columns))
try:
    display(nuts_sample.head())
except Exception:
    pass

try:
    counties_sample = gpd.read_file(US_COUNTIES_PATH, rows=5)
except TypeError:
    counties_sample = gpd.read_file(US_COUNTIES_PATH)
print("\nColumns in US counties GeoJSON:")
print(list(counties_sample.columns))
try:
    display(counties_sample.head())
except Exception:
    pass


def inspect_gadm2(gpkg_path: str, layer: str = "gadm_410", prefer_code_cols=("GID_2", "ID_2", "GID2")):
    """
    Inspect the GADM layer and return:
      - full GeoDataFrame
      - ADM2 subset
      - selected code column name
    """
    gdf = gpd.read_file(gpkg_path, layer=layer)
    print(f"[INFO] Loaded layer='{layer}' with {len(gdf):,} features")
    print(f"[INFO] CRS: {gdf.crs}")
    print("[INFO] Columns:")
    print(list(gdf.columns))

    cols_up = {c.upper(): c for c in gdf.columns}
    code_col = None
    for pref in prefer_code_cols:
        if pref in cols_up:
            code_col = cols_up[pref]
            break
    if code_col is None:
        candidates = [orig for up, orig in cols_up.items() if up.endswith("_2") or "GID" in up]
        if not candidates:
            raise ValueError("Missing ADM2 code column (e.g., 'GID_2' or 'ID_2').")
        code_col = candidates[0]

    print(f"[INFO] ADM2 code column: {code_col!r}")
    gdf_adm2 = gdf[~gdf[code_col].isna()].copy()
    print(f"[INFO] ADM2 rows (non-NaN in {code_col}): {len(gdf_adm2):,}")

    unique_codes = gdf_adm2[code_col].astype("string").str.strip().str.upper().nunique()
    print(f"[INFO] Unique ADM2 codes: {unique_codes:,}")

    geom_types = gdf_adm2.geom_type.value_counts().to_dict()
    print(f"[INFO] Geometry types: {geom_types}")

    candidates_name = [c for c in ["NAME_0", "NAME_1", "NAME_2", "GID_0", "GID_1", "GID_2"] if c in gdf_adm2.columns]
    show_cols = [code_col] + candidates_name
    show_cols = [c for c in dict.fromkeys(show_cols)]

    print("\n== ADM2 preview (head) ==")
    try:
        display(gdf_adm2[show_cols].head())
    except Exception:
        pass

    if any(c in gdf_adm2.columns for c in ["NAME_0", "NAME_1", "NAME_2", "GID_0", "GID_1", "GID_2"]):
        print("\n== Missing counts on common columns ==")
        check_cols = [c for c in ["NAME_0", "NAME_1", "NAME_2", "GID_0", "GID_1", "GID_2"] if c in gdf_adm2.columns]
        try:
            display(gdf_adm2[check_cols].isna().sum().to_frame("missing"))
        except Exception:
            pass

    return gdf, gdf_adm2, code_col


# Run quick inspection once (useful the first time)
gdf_full, gdf_adm2, gadm_code_col = inspect_gadm2(GADM_GPKG_PATH, layer="gadm_410")


# ==========================
# Nodes pipeline
# ==========================

# 0) Restrict mapping to codes present in SCI for selected levels
target = select_target_codes(
    df_sci=df_sci,
    df_map=df_map,
    level_types=("NUTS3", "GADM2", "GADM1", "COUNTY")
)

# 1) Load representative points per dataset
nuts_pts = load_nuts3_points(NUTS_GEOJSON_PATH)
county_pts = load_us_counties_points(US_COUNTIES_PATH)
gadm1_pts = load_gadm1_points(GADM_GPKG_PATH, layer="gadm_410", code_col="GID_1")
gadm2_pts = load_gadm2_points(GADM_GPKG_PATH, layer="gadm_410", code_col="GID_2")

print("NUTS pts:", len(nuts_pts), "— US counties pts:", len(county_pts),
      "— GADM1 pts:", len(gadm1_pts), "— GADM2 pts:", len(gadm2_pts))

# 2) Join mapping to geodata (with code normalization where needed)
nuts_nodes = build_nodes_for_type_with_transform(target, "NUTS3", nuts_pts, transform_fn=None)
gadm2_nodes = build_nodes_for_type_with_transform(target, "GADM2", gadm2_pts, transform_fn=normalize_gadm2_mapping_code)
gadm1_nodes = build_nodes_for_type_with_transform(target, "GADM1", gadm1_pts, transform_fn=normalize_gadm1_mapping_code)
county_nodes = build_nodes_for_type_with_transform(target, "COUNTY", county_pts, transform_fn=normalize_county_mapping_code)

print("Selected NUTS3:", len(nuts_nodes))
print("Selected GADM2:", len(gadm2_nodes))
print("Selected GADM1 (before filter):", len(gadm1_nodes))
print("Selected COUNTY:", len(county_nodes))

# 3) Assemble final node list with priority GADM2 > GADM1
node_list = assemble_node_list(nuts_nodes, gadm2_nodes, gadm1_nodes, county_nodes)

try:
    display(node_list.head())
except Exception:
    pass
print(f"Total nodes: {len(node_list):,}")

# 4) Save nodes
OUTPUT_CSV = "node_list.csv"
node_list.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV} (columns: {list(node_list.columns)})")


# ==========================
# Edge list: general and per-country
# ==========================

def build_nuts_a2_to_iso3_map(nuts_geojson_path: str) -> Dict[str, str]:
    """
    Build a mapping from NUTS alpha-2 country code to ISO3.
    Uses pycountry if available, otherwise handles known exceptions.
    """
    gdf = gpd.read_file(nuts_geojson_path)
    if "CNTR_CODE" not in gdf.columns:
        raise ValueError("Missing 'CNTR_CODE' in NUTS GeoJSON.")
    a2_vals = gdf["CNTR_CODE"].dropna().astype(str).str.upper().unique().tolist()
    mapping: Dict[str, str] = {}
    try:
        import pycountry
        for a2 in a2_vals:
            a2_norm = "GB" if a2 == "UK" else ("GR" if a2 == "EL" else a2)
            c = pycountry.countries.get(alpha_2=a2_norm)
            if c and hasattr(c, "alpha_3"):
                mapping[a2] = c.alpha_3
    except Exception:
        repl = {"UK": "GBR", "EL": "GRC"}
        for a2 in a2_vals:
            if a2 in repl:
                mapping[a2] = repl[a2]
    if not mapping:
        print("[WARN] Could not build NUTS alpha2 -> ISO3 map. Consider installing 'pycountry'.")
    return mapping


NUTS_A2_TO_ISO3: Dict[str, str] = build_nuts_a2_to_iso3_map(NUTS_GEOJSON_PATH)


def iso3_from_node_label(nodeLabel: str, dataset: str) -> Optional[str]:
    """Infer ISO3 from nodeLabel and dataset type."""
    if not isinstance(nodeLabel, str):
        return None
    ds = (dataset or "").upper()
    if ds in ("GADM1", "GADM2"):
        m = re.match(r"^([A-Z]{3})\.", nodeLabel)
        return m.group(1) if m else None
    if ds == "COUNTY":
        return "USA"
    if ds == "NUTS3":
        a2 = nodeLabel[:2].upper()
        return NUTS_A2_TO_ISO3.get(a2)
    return None


def country_name_from_iso3(iso3: Optional[str]) -> Optional[str]:
    """Return a readable country name from ISO3, when pycountry is available."""
    if iso3 is None or pd.isna(iso3):
        return None
    try:
        import pycountry
        c = pycountry.countries.get(alpha_3=str(iso3))
        if c:
            return getattr(c, "common_name", None) or getattr(c, "name", None)
    except Exception:
        pass
    return None


def _detect_country_cols(df_map: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Look for country ISO3 and name columns in the mapping table."""
    iso_candidates = ["country_ISO3", "ISO3", "iso3", "COUNTRY_ISO3", "country_iso3"]
    name_candidates = ["country_name", "COUNTRY_NAME", "country", "COUNTRY", "NAME_0"]
    iso_col = next((c for c in iso_candidates if c in df_map.columns), None)
    name_col = next((c for c in name_candidates if c in df_map.columns), None)
    return iso_col, name_col


def build_loc2node_mapping(
    nodes: pd.DataFrame,
    df_sci: pd.DataFrame,
    df_map: pd.DataFrame,
    allowed_datasets: Iterable[str] = ("NUTS3", "GADM2", "GADM1", "COUNTY")
) -> pd.DataFrame:
    """
    Build the mapping:
      location_code (SCI) -> (nodeID, nodeLabel, DatasetDiOrigine, country_ISO3, country_name)
    Country fields are taken from mapping when available; otherwise inferred.
    """
    allowed = {t.upper() for t in allowed_datasets}
    nodes_ok = nodes[nodes["DatasetDiOrigine"].str.upper().isin(allowed)].copy()

    target = select_target_codes(df_sci=df_sci, df_map=df_map, level_types=allowed)

    parts = []
    if "NUTS3" in allowed:
        t = target[target["level_type"].str.upper() == "NUTS3"].copy()
        t["code"] = t["location_code"]
        t["nodeLabel"] = t["code"]
        parts.append(t)
    if "GADM2" in allowed:
        t = target[target["level_type"].str.upper() == "GADM2"].copy()
        t["code"] = t["location_code"].map(normalize_gadm2_mapping_code)
        t["nodeLabel"] = t["code"]
        parts.append(t)
    if "GADM1" in allowed:
        t = target[target["level_type"].str.upper() == "GADM1"].copy()
        t["code"] = t["location_code"].map(normalize_gadm1_mapping_code)
        t["nodeLabel"] = t["code"]
        parts.append(t)
    if "COUNTY" in allowed:
        t = target[target["level_type"].str.upper() == "COUNTY"].copy()
        t["code"] = t["location_code"].map(normalize_county_mapping_code)
        t["nodeLabel"] = "USA" + t["code"].astype(str).str.zfill(5)
        parts.append(t)

    if not parts:
        raise ValueError("No allowed datasets to build loc->node mapping.")

    mapping_raw = (pd.concat(parts, ignore_index=True)
                   .dropna(subset=["nodeLabel"])
                   .drop_duplicates(["location_code", "nodeLabel"]))

    loc2node = mapping_raw.merge(
        nodes_ok[["nodeID", "nodeLabel", "DatasetDiOrigine"]],
        on="nodeLabel", how="inner"
    )

    iso_col, name_col = _detect_country_cols(df_map)
    if iso_col is not None:
        loc2country = df_map[["location_code", iso_col]].copy().rename(columns={iso_col: "country_ISO3"})
        if name_col and name_col in df_map.columns and name_col != iso_col:
            loc2country["country_name"] = df_map[name_col]
        else:
            loc2country["country_name"] = pd.NA
        loc2node = loc2node.merge(loc2country, on="location_code", how="left")
    else:
        loc2node["country_ISO3"] = pd.NA
        loc2node["country_name"] = pd.NA

    miss_iso = loc2node["country_ISO3"].isna()
    if miss_iso.any():
        loc2node.loc[miss_iso, "country_ISO3"] = loc2node.loc[miss_iso].apply(
            lambda r: iso3_from_node_label(r["nodeLabel"], r["DatasetDiOrigine"]), axis=1
        ).astype("string")

    miss_name = loc2node["country_name"].isna() if "country_name" in loc2node.columns else pd.Series(True, index=loc2node.index)
    if miss_name.any():
        loc2node.loc[miss_name, "country_name"] = loc2node.loc[miss_name, "country_ISO3"].map(country_name_from_iso3)

    loc2node = (loc2node
                .sort_values(["location_code", "DatasetDiOrigine"])
                .drop_duplicates("location_code"))

    assert loc2node["nodeID"].isna().sum() == 0, "loc->node mapping contains missing nodeID"
    return loc2node[["location_code", "nodeID", "nodeLabel", "DatasetDiOrigine", "country_ISO3", "country_name"]]


def build_edge_lists(
    df_sci: pd.DataFrame,
    df_map: pd.DataFrame,
    nodes: pd.DataFrame,
    allowed_datasets: Iterable[str] = ("NUTS3", "GADM2", "GADM1", "COUNTY"),
    output_dir: str = ".",
    general_filename: str = "edges_all.csv",
    per_country: bool = True,
    country_column_mode: str = "from"  # 'from' | 'to' | 'common_or_null'
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Build and save edges:
      - edges_all.csv with (nodeID_from, nodeID_to, country_name, country_ISO3, weight)
      - edges_by_country/edges_<ISO3>.csv for intra-country edges, if requested
    """
    os.makedirs(output_dir, exist_ok=True)

    loc2node = build_loc2node_mapping(nodes, df_sci, df_map, allowed_datasets)

    sci = df_sci[["user_loc", "fr_loc", "scaled_sci"]].copy()
    sci["user_loc"] = _normalize_location_code(sci["user_loc"])
    sci["fr_loc"] = _normalize_location_code(sci["fr_loc"])

    left = loc2node.add_prefix("from_")
    right = loc2node.add_prefix("to_")

    e = (sci
         .merge(left, left_on="user_loc", right_on="from_location_code", how="inner")
         .merge(right, left_on="fr_loc", right_on="to_location_code", how="inner"))

    if country_column_mode == "to":
        country_iso = e["to_country_ISO3"]
        country_name = e["to_country_name"]
    elif country_column_mode == "common_or_null":
        same = e["from_country_ISO3"].notna() & (e["from_country_ISO3"] == e["to_country_ISO3"])
        country_iso = e["from_country_ISO3"].where(same)
        country_name = e["from_country_name"].where(same)
    else:
        country_iso = e["from_country_ISO3"]
        country_name = e["from_country_name"]

    edges_all = pd.DataFrame({
        "nodeID_from": e["from_nodeID"].astype(int),
        "nodeID_to": e["to_nodeID"].astype(int),
        "country_name": country_name.astype("string"),
        "country_ISO3": country_iso.astype("string"),
        "weight": e["scaled_sci"].astype(float),
    })

    out_all = os.path.join(output_dir, general_filename)
    edges_all.to_csv(out_all, index=False)
    print(f"Saved edges (all): {out_all} (rows: {len(edges_all):,})")

    edges_by_iso: Dict[str, pd.DataFrame] = {}
    if per_country:
        intra_mask = e["from_country_ISO3"].notna() & (e["from_country_ISO3"] == e["to_country_ISO3"])
        intra = e.loc[intra_mask].copy()
        if len(intra):
            by_dir = os.path.join(output_dir, "edges_by_country")
            os.makedirs(by_dir, exist_ok=True)
            intra["country_ISO3"] = intra["from_country_ISO3"].astype(str)
            intra["country_name"] = intra["from_country_name"]

            for iso3, grp in intra.groupby("country_ISO3"):
                df_iso = pd.DataFrame({
                    "nodeID_from": grp["from_nodeID"].astype(int),
                    "nodeID_to": grp["to_nodeID"].astype(int),
                    "country_name": grp["country_name"].astype("string"),
                    "country_ISO3": grp["country_ISO3"].astype("string"),
                    "weight": grp["scaled_sci"].astype(float)
                })
                edges_by_iso[iso3] = df_iso
                file_iso = os.path.join(by_dir, f"edges_{iso3}.csv")
                df_iso.to_csv(file_iso, index=False)
            print(f"Saved {len(edges_by_iso)} intra-country files in: {by_dir}")
        else:
            print("[INFO] No intra-country edges found with current filters.")

    return edges_all, edges_by_iso


# Build edge lists
ALLOWED_DATASETS = ("NUTS3", "GADM2", "GADM1", "COUNTY")

df_edges_all, edges_per_country = build_edge_lists(
    df_sci=df_sci,
    df_map=df_map,
    nodes=node_list,
    allowed_datasets=ALLOWED_DATASETS,
    output_dir=".",
    general_filename="edges_all.csv",
    per_country=True,
    country_column_mode="from"
)


# ==========================
# Coverage summary of FB nodes against built nodes
# ==========================

def summarize_fb_node_coverage(
    df_sci: pd.DataFrame,
    df_map: pd.DataFrame,
    node_list: pd.DataFrame,
    allowed_datasets: Iterable[str] = ("NUTS3", "GADM2", "GADM1", "COUNTY"),
    by_country: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Summarize coverage of FB nodes (SCI codes present in mapping) vs built node list:
      - by dataset type
      - optionally by country within each dataset
    """
    def _detect_country_cols_local(df_map: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        iso_candidates = ["country_ISO3", "ISO3", "iso3", "COUNTRY_ISO3", "country_iso3"]
        name_candidates = ["country_name", "COUNTRY_NAME", "country", "COUNTRY", "NAME_0"]
        iso_col = next((c for c in iso_candidates if c in df_map.columns), None)
        name_col = next((c for c in name_candidates if c in df_map.columns), None)
        return iso_col, name_col

    def _node_label_from_mapping_row(level_type: str, location_code: str) -> Optional[str]:
        lt = (level_type or "").upper()
        if pd.isna(location_code):
            return None
        code = str(location_code).strip().str.upper()
        if lt == "NUTS3":
            return code
        if lt == "GADM1":
            return normalize_gadm1_mapping_code(code)
        if lt == "GADM2":
            return normalize_gadm2_mapping_code(code)
        if lt == "COUNTY":
            c = normalize_county_mapping_code(code)
            return None if c is None else ("USA" + str(c).zfill(5))
        return None

    allowed = {t.upper() for t in allowed_datasets}

    sci_codes = set(get_unique_location_codes(df_sci))

    labels_by_ds: Dict[str, set[str]] = {
        ds: set(node_list.loc[node_list["DatasetDiOrigine"].str.upper() == ds, "nodeLabel"].astype(str))
        for ds in ["NUTS3", "GADM2", "GADM1", "COUNTY"]
    }

    iso_col, name_col = _detect_country_cols_local(df_map)

    rows_type: List[Dict] = []
    rows_country: List[Dict] = []

    for ds in ["NUTS3", "GADM2", "GADM1", "COUNTY"]:
        if ds not in allowed:
            continue

        mm = df_map[df_map["level_type"].str.upper() == ds].copy()
        mm = mm[mm["location_code"].isin(sci_codes)].copy()

        mm["nodeLabel"] = mm.apply(lambda r: _node_label_from_mapping_row(ds, r["location_code"]), axis=1)
        mm = mm.dropna(subset=["nodeLabel"]).drop_duplicates("location_code")

        fb_unique = mm["location_code"].nunique()
        matched = int(mm["nodeLabel"].isin(labels_by_ds.get(ds, set())).sum())

        rows_type.append({
            "dataset_type": ds,
            "fb_unique_nodes": int(fb_unique),
            "matched_nodes": matched,
            "coverage_pct": (matched / fb_unique * 100.0) if fb_unique else 0.0,
            "unmatched_nodes": int(fb_unique - matched),
        })

        if by_country and iso_col is not None:
            cols = ["location_code", iso_col]
            if name_col and name_col in df_map.columns and name_col != iso_col:
                cols.append(name_col)
            tmp = mm.merge(df_map[cols], on="location_code", how="left").rename(columns={iso_col: "country_ISO3"})
            if name_col and name_col in tmp.columns and name_col != iso_col:
                tmp = tmp.rename(columns={name_col: "country_name"})
            else:
                tmp["country_name"] = pd.NA
            tmp["is_matched"] = tmp["nodeLabel"].isin(labels_by_ds.get(ds, set()))

            for (iso3, cname), grp in tmp.groupby(["country_ISO3", "country_name"], dropna=False):
                fb_u = grp["location_code"].nunique()
                m_u = int(grp["is_matched"].sum())
                rows_country.append({
                    "country_ISO3": iso3,
                    "country_name": cname,
                    "dataset_type": ds,
                    "fb_unique_nodes": int(fb_u),
                    "matched_nodes": m_u,
                    "coverage_pct": (m_u / fb_u * 100.0) if fb_u else 0.0,
                    "unmatched_nodes": int(fb_u - m_u),
                })

    df_type = pd.DataFrame(rows_type).sort_values(["dataset_type"]).reset_index(drop=True)
    df_country = (pd.DataFrame(rows_country)
                  .sort_values(["dataset_type", "country_ISO3"])
                  .reset_index(drop=True)) if rows_country else None

    print("\n== FB → NODES COVERAGE by TYPE ==")
    try:
        display(df_type)
    except Exception:
        print(df_type)

    if df_country is not None:
        print("\n== FB → NODES COVERAGE by COUNTRY × TYPE (head) ==")
        try:
            display(df_country.head(20))
        except Exception:
            print(df_country.head(20))

    return df_type, df_country


coverage_by_type, coverage_by_country = summarize_fb_node_coverage(
    df_sci=df_sci,
    df_map=df_map,
    node_list=node_list
)


# ==========================
# Cache useful artifacts
# ==========================
os.makedirs("cache", exist_ok=True)

names = [
    "df_sci", "df_map",
    "nuts_pts", "gadm2_pts", "gadm1_pts", "county_pts",
    "target",
    "nuts_nodes", "gadm2_nodes", "gadm1_nodes", "county_nodes",
    "node_list",
    "df_edges_all", "edges_per_country",
    "coverage_by_type", "coverage_by_country"
]

for n in names:
    if n in globals():
        pd.to_pickle(globals()[n], f"cache/{n}.pkl")
        print(f"Saved {n} → cache/{n}.pkl")
    else:
        print(f"[WARN] Variable not found: {n}")

# Node label -> nodeID map
if "node_list" in globals():
    node_id_map = node_list.set_index("nodeLabel")["nodeID"].to_dict()
    pd.to_pickle(node_id_map, "cache/node_id_map.pkl")
    print("Saved node_id_map → cache/node_id_map.pkl")

# Mapping location_code (SCI) -> node, for reuse without rejoining
try:
    ALLOWED_DATASETS = ("NUTS3", "GADM2", "GADM1", "COUNTY")
    loc2node = build_loc2node_mapping(node_list, df_sci, df_map, ALLOWED_DATASETS)
    pd.to_pickle(loc2node, "cache/loc2node.pkl")
    print("Saved loc2node → cache/loc2node.pkl")
except Exception as e:
    print(f"[WARN] loc2node not saved: {e}")
