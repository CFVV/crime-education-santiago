"""
06_spatial_buffers.py
======================
Construye los índices de exposición al crimen por escuela,
para múltiples ventanas temporales y radios espaciales.

Arquitectura:
    Para cada combinación (rbd, agno, grado):
        1. Obtiene la fecha exacta del examen desde data_education_2013_2019_RM
           (columna fecha_rendicion — una fecha por agno-grado)
        2. Selecciona crímenes en [fecha - N días, fecha] dentro del buffer
        3. Calcula indice_exp ponderado por distancia inversa
        4. Repite para cada ventana (7, 14, 30, 90, 180, 365 días)
           y cada radio (400m, 1km)

Inputs:
    - crime_index_geocode_daily.csv         ← Script 03
    - crime_detail_geocode_daily.csv        ← Script 03
    - school_locations_RM.parquet           ← Script 04
    - data_education_2013_2019_RM.parquet   ← Script 05 (fechas rendición)
    - rm_manzanas_geocodigo.shp             ← shapefile INE

Outputs:
    - crime_exposure_400m.parquet   → rbd × agno × grado × ventana_dias
    - crime_exposure_1km.parquet    → ídem
    - crime_index_georef.parquet    → manzana × día con lat/lon
    - crime_detail_georef.parquet   → manzana × día × delito con lat/lon

Columnas output (por fila rbd × agno × grado × ventana_dias):
    rbd, agno, grado, ventana_dias, fecha_examen,
    indice_exp, frecuencia_total, n_manzanas,
    n_nivel_1 … n_nivel_8

Fórmula índice de exposición (decay por distancia):
    peso_m = 1 / (1 + dist_metros)
    indice_exp = sum(indice_m × peso_m × frec_m) / sum(peso_m × frec_m)

Ejecución:
    python scripts/06_spatial_buffers.py

Dependencias:
    pip install geopandas shapely pyproj
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import (
    CENSUS_BLOCKS_PATH,
    CRIME_BUFFER_400M_PATH,
    CRIME_BUFFER_1KM_PATH,
    CRIME_DETAIL_DAILY_PATH,
    CRIME_DETAIL_GEOREF_PATH,
    CRIME_GEOREF_PATH,
    CRIME_INDEX_DAILY_PATH,
    EDUCATION_OUTPUT_PATH,
    SCHOOL_LOCATIONS_PATH,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BUFFER_RADII     = [400, 1000]
TEMPORAL_WINDOWS = [7, 14, 30, 90, 180, 365]
CRS_PROJECTED    = "EPSG:32719"   # UTM zona 19S — metros
CRS_WGS84        = "EPSG:4326"
NIVELES          = list(range(1, 9))

BUFFER_OUTPUT_PATHS = {
    400:  CRIME_BUFFER_400M_PATH,
    1000: CRIME_BUFFER_1KM_PATH,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def guardar_parquet(df: pd.DataFrame, path: Path, descripcion: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Guardado: %s — %d filas × %d cols  [%s]",
             path.name, len(df), len(df.columns), descripcion)


def _fila_cero(rbd, agno, grado, ventana, fecha_str):
    row = {
        "rbd":              int(rbd),
        "agno":             agno,
        "grado":            grado,
        "ventana_dias":     ventana,
        "fecha_examen":     fecha_str,
        "indice_exp":       0.0,
        "frecuencia_total": 0,
        "n_manzanas":       0,
    }
    for n in NIVELES:
        row[f"n_nivel_{n}"] = 0
    return row


# ---------------------------------------------------------------------------
# Paso 1 — Fechas SIMCE exactas desde data_education
# ---------------------------------------------------------------------------

def cargar_fechas_simce() -> dict:
    """
    Carga fecha_rendicion desde data_education_2013_2019_RM.parquet.
    Retorna dict: {(agno, grado): fecha_str} — una fecha por combinación.
    """
    if not EDUCATION_OUTPUT_PATH.exists():
        log.error("No se encontró: %s", EDUCATION_OUTPUT_PATH)
        log.error("Corre primero: python scripts/05_dataprep_education.py")
        sys.exit(1)

    df = pd.read_parquet(EDUCATION_OUTPUT_PATH, columns=["agno", "grado", "fecha_rendicion"])
    df = df.dropna(subset=["fecha_rendicion"])

    fechas = (
        df.groupby(["agno", "grado"])["fecha_rendicion"]
        .first()
        .reset_index()
    )

    fecha_dict = {
        (int(row.agno), str(row.grado)): str(row.fecha_rendicion)[:10]
        for row in fechas.itertuples()
    }

    log.info("Fechas SIMCE cargadas: %d combinaciones agno-grado", len(fecha_dict))
    for (agno, grado), fecha in sorted(fecha_dict.items()):
        log.info("  %d %s → %s", agno, grado, fecha)

    return fecha_dict


# ---------------------------------------------------------------------------
# Paso 2 — Shapefile manzanas
# ---------------------------------------------------------------------------

def cargar_manzanas():
    try:
        import geopandas as gpd
    except ImportError:
        log.error("geopandas no instalado.")
        sys.exit(1)

    if not CENSUS_BLOCKS_PATH.exists():
        log.error("No se encontró: %s", CENSUS_BLOCKS_PATH)
        sys.exit(1)

    log.info("Cargando shapefile: %s", CENSUS_BLOCKS_PATH.name)
    gdf = gpd.read_file(CENSUS_BLOCKS_PATH, encoding="latin-1")
    log.info("Shapefile: %d sub-manzanas | CRS: %s", len(gdf), gdf.crs)

    gdf["GEOCODIGO_11"] = gdf["GEOCODIGO"].astype(str).str.strip().str[:11]
    log.info("Disolviendo sub-manzanas...")
    gdf_mz = gdf.dissolve(by="GEOCODIGO_11", as_index=False)[
        ["GEOCODIGO_11", "COMUNA", "geometry"]
    ]
    log.info("Manzanas disueltas: %d", len(gdf_mz))

    if gdf_mz.crs is None:
        gdf_mz = gdf_mz.set_crs(CRS_WGS84)
    gdf_mz = gdf_mz.to_crs(CRS_PROJECTED)

    centroid     = gdf_mz.geometry.centroid
    centroid_wgs = centroid.to_crs(CRS_WGS84)
    gdf_mz["lat_mz"] = centroid_wgs.y
    gdf_mz["lon_mz"] = centroid_wgs.x
    gdf_mz["cx"]     = centroid.x
    gdf_mz["cy"]     = centroid.y

    return gdf_mz


# ---------------------------------------------------------------------------
# Paso 3 — Índice crimen diario
# ---------------------------------------------------------------------------

def cargar_indice_diario() -> pd.DataFrame:
    if not CRIME_INDEX_DAILY_PATH.exists():
        log.error("No se encontró: %s", CRIME_INDEX_DAILY_PATH)
        sys.exit(1)

    df = pd.read_csv(CRIME_INDEX_DAILY_PATH, low_memory=False)
    df["GEOCODIGO"] = df["GEOCODIGO"].astype(str).str.strip().str.zfill(11)
    df["fecha"]     = pd.to_datetime(df["dia"], errors="coerce")
    log.info("Índice diario: %d filas | GEOCODIGOs: %d | rango: %s → %s",
             len(df), df["GEOCODIGO"].nunique(),
             df["fecha"].min().date(), df["fecha"].max().date())
    return df


# ---------------------------------------------------------------------------
# Paso 4 — Escuelas
# ---------------------------------------------------------------------------

def cargar_escuelas():
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit(1)

    schools = pd.read_parquet(SCHOOL_LOCATIONS_PATH).dropna(
        subset=["latitud", "longitud"]
    )
    gdf = gpd.GeoDataFrame(
        schools,
        geometry=gpd.points_from_xy(schools["longitud"], schools["latitud"]),
        crs=CRS_WGS84,
    ).to_crs(CRS_PROJECTED)

    gdf["sx"] = gdf.geometry.x
    gdf["sy"] = gdf.geometry.y
    log.info("Escuelas: %d RBDs con coordenadas", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Paso 5 — Join manzanas × crimen
# ---------------------------------------------------------------------------

def unir_manzanas_con_crimen(gdf_mz, df_crimen) -> pd.DataFrame:
    meta   = gdf_mz[["GEOCODIGO_11", "cx", "cy"]].rename(
        columns={"GEOCODIGO_11": "GEOCODIGO"}
    )
    merged = df_crimen.merge(meta, on="GEOCODIGO", how="inner")
    n = df_crimen["GEOCODIGO"].nunique()
    m = merged["GEOCODIGO"].nunique()
    log.info("Match GEOCODIGOs: %d / %d (%.1f%%)", m, n, 100 * m / n)
    return merged


# ---------------------------------------------------------------------------
# Paso 6 — Calcular exposición por escuela × agno × grado × ventana × radio
# ---------------------------------------------------------------------------

def calcular_exposicion(
    df_crimen_geo: pd.DataFrame,
    gdf_schools,
    radio_m: int,
    fechas_simce: dict,
) -> pd.DataFrame:

    log.info("Buffer %dm: %d combos agno-grado × %d ventanas × %d escuelas",
             radio_m, len(fechas_simce), len(TEMPORAL_WINDOWS), len(gdf_schools))

    school_arr = gdf_schools[["rbd", "sx", "sy"]].values
    resultados = []
    total      = len(fechas_simce)
    ventana_max = max(TEMPORAL_WINDOWS)

    for idx, ((agno, grado), fecha_str) in enumerate(sorted(fechas_simce.items())):
        if idx % 5 == 0:
            log.info("  %d/%d — %d %s (%s)", idx + 1, total, agno, grado, fecha_str)

        fecha_examen     = pd.Timestamp(fecha_str)
        fecha_inicio_max = fecha_examen - pd.Timedelta(days=ventana_max)

        # Pre-filtrar crimen en ventana máxima (optimización)
        df_pre = df_crimen_geo[
            (df_crimen_geo["fecha"] >= fecha_inicio_max) &
            (df_crimen_geo["fecha"] <= fecha_examen)
        ]

        if df_pre.empty:
            for ventana in TEMPORAL_WINDOWS:
                for rbd, sx, sy in school_arr:
                    resultados.append(_fila_cero(rbd, agno, grado, ventana, fecha_str))
            continue

        mz_xy_all  = df_pre[["cx", "cy"]].values
        fechas_all = df_pre["fecha"].values
        indice_all = df_pre["indice_delictualidad"].values
        frec_all   = df_pre["frecuencia_total"].values
        nivel_all  = {n: df_pre[f"n_nivel_{n}"].values
                      for n in NIVELES if f"n_nivel_{n}" in df_pre.columns}

        for ventana in TEMPORAL_WINDOWS:
            fecha_inicio_v = fecha_examen - pd.Timedelta(days=ventana)
            mask_t = fechas_all >= np.datetime64(fecha_inicio_v)

            mz_xy_v  = mz_xy_all[mask_t]
            indice_v = indice_all[mask_t]
            frec_v   = frec_all[mask_t]
            nivel_v  = {n: v[mask_t] for n, v in nivel_all.items()}

            if len(mz_xy_v) == 0:
                for rbd, sx, sy in school_arr:
                    resultados.append(_fila_cero(rbd, agno, grado, ventana, fecha_str))
                continue

            for rbd, sx, sy in school_arr:
                dists  = np.sqrt((mz_xy_v[:, 0] - sx)**2 +
                                 (mz_xy_v[:, 1] - sy)**2)
                mask_r = dists <= radio_m

                if not mask_r.any():
                    resultados.append(_fila_cero(rbd, agno, grado, ventana, fecha_str))
                    continue

                d_in    = dists[mask_r]
                idx_in  = indice_v[mask_r]
                frec_in = frec_v[mask_r]
                pesos   = 1 / (1 + d_in)
                denom   = (pesos * frec_in).sum()

                row = {
                    "rbd":              int(rbd),
                    "agno":             agno,
                    "grado":            grado,
                    "ventana_dias":     ventana,
                    "fecha_examen":     fecha_str,
                    "indice_exp":       round((idx_in * pesos * frec_in).sum() / denom, 6)
                                        if denom > 0 else 0.0,
                    "frecuencia_total": int(frec_in.sum()),
                    "n_manzanas":       int(mask_r.sum()),
                }
                for n in NIVELES:
                    row[f"n_nivel_{n}"] = int(nivel_v[n][mask_r].sum()) if n in nivel_v else 0

                resultados.append(row)

    df_out = pd.DataFrame(resultados)
    log.info("Buffer %dm completado: %d filas", radio_m, len(df_out))
    return df_out


# ---------------------------------------------------------------------------
# Archivos georeferenciados (visualización)
# ---------------------------------------------------------------------------

def guardar_crimen_georef(df_crimen_geo, gdf_mz):
    meta = gdf_mz[["GEOCODIGO_11", "lat_mz", "lon_mz", "COMUNA"]].rename(
        columns={"GEOCODIGO_11": "GEOCODIGO", "lat_mz": "lat",
                 "lon_mz": "lon", "COMUNA": "comuna"}
    )
    df = df_crimen_geo.merge(meta, on="GEOCODIGO", how="left")
    cols = (["GEOCODIGO", "fecha", "anio", "dia", "indice_delictualidad",
              "frecuencia_total"] +
            [c for c in df.columns if c.startswith("n_nivel_")] +
            ["lat", "lon", "comuna"])
    df = df[[c for c in cols if c in df.columns]].sort_values(["GEOCODIGO", "fecha"])
    guardar_parquet(df, CRIME_GEOREF_PATH, "crimen georeferenciado diario")
    log.info("  → %d GEOCODIGOs | rango: %s → %s",
             df["GEOCODIGO"].nunique(),
             df["fecha"].min().date(), df["fecha"].max().date())


def guardar_crimen_detail_georef(gdf_mz):
    if not CRIME_DETAIL_DAILY_PATH.exists():
        log.warning("No se encontró %s — omitiendo", CRIME_DETAIL_DAILY_PATH.name)
        return

    log.info("Cargando crimen detallado diario...")
    df = pd.read_csv(CRIME_DETAIL_DAILY_PATH, low_memory=False)
    df["GEOCODIGO"] = df["GEOCODIGO"].astype(str).str.strip().str.zfill(11)
    df["fecha"]     = pd.to_datetime(df["dia"], errors="coerce")
    meta = gdf_mz[["GEOCODIGO_11", "lat_mz", "lon_mz", "COMUNA"]].rename(
        columns={"GEOCODIGO_11": "GEOCODIGO", "lat_mz": "lat",
                 "lon_mz": "lon", "COMUNA": "comuna"}
    )
    df = df.merge(meta, on="GEOCODIGO", how="left")
    n_sin = df["lat"].isna().sum()
    if n_sin:
        log.warning("%d filas sin geometría (%.1f%%)", n_sin, 100 * n_sin / len(df))
    df = df.sort_values(["GEOCODIGO", "fecha", "delito"])
    guardar_parquet(df, CRIME_DETAIL_GEOREF_PATH,
                    "crimen detallado georeferenciado diario")
    log.info("  → %d GEOCODIGOs | %d delitos | %d filas",
             df["GEOCODIGO"].nunique(), df["delito"].nunique(), len(df))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Script 06: Buffers espaciales crimen-escuelas ===")
    log.info("Radios: %s m | Ventanas: %s días", BUFFER_RADII, TEMPORAL_WINDOWS)

    try:
        import geopandas  # noqa
    except ImportError:
        log.error("Instala: pip install geopandas shapely pyproj")
        sys.exit(1)

    # 1. Fechas exactas SIMCE
    fechas_simce = cargar_fechas_simce()

    # 2. Shapefile manzanas
    gdf_mz = cargar_manzanas()

    # 3. Índice crimen diario
    df_crimen = cargar_indice_diario()

    # 4. Escuelas
    gdf_schools = cargar_escuelas()

    # 5. Join manzanas × crimen
    df_cg = unir_manzanas_con_crimen(gdf_mz, df_crimen)

    # 6. Archivos georeferenciados
    log.info("── Archivos georeferenciados ──")
    guardar_crimen_georef(df_cg, gdf_mz)
    guardar_crimen_detail_georef(gdf_mz)

    # 7. Buffers
    for radio in BUFFER_RADII:
        log.info("── Buffer %dm ──", radio)
        df_exp = calcular_exposicion(df_cg, gdf_schools, radio, fechas_simce)
        guardar_parquet(df_exp, BUFFER_OUTPUT_PATHS[radio],
                        f"exposición crimen buffer {radio}m")

    # 8. Resumen
    log.info("── Resumen final ──")
    for radio in BUFFER_RADII:
        df = pd.read_parquet(BUFFER_OUTPUT_PATHS[radio])
        log.info("Buffer %dm: %d filas | RBDs: %d | agno-grado: %d | ventanas: %s",
                 radio, len(df), df["rbd"].nunique(),
                 df[["agno", "grado"]].drop_duplicates().shape[0],
                 sorted(df["ventana_dias"].unique().tolist()))

    log.info("=== Script 06 completado ===")


if __name__ == "__main__":
    main()
