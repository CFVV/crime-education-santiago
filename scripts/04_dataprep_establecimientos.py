"""
04_dataprep_establecimientos.py
================================
Prepara el panel completo de establecimientos educacionales 2013–2019 RM.

Tareas:
    1. Carga los 7 directorios anuales MINEDUC (2013–2019) desde data/raw/
    2. Normaliza columnas, tipos y encoding (maneja formatos inconsistentes)
    3. Imputa lat/lon del año 2013 copiando coordenadas de 2014
       (2013 no tiene lat/lon en el directorio original)
    4. Valida coordenadas: rango de Chile, estabilidad año a año
    5. Calcula coordenada madre por RBD (moda sobre coords redondeadas a 4 decimales)
    6. Reemplaza lat/lon anuales por coordenada madre
       → elimina ruido de coordenadas erróneas año a año
    7. Filtra solo Región Metropolitana (cod_reg_rbd == 13)
    8. Guarda:
         - schools_clean_2013_2019_RM.parquet  ← panel completo con coord madre, solo RM
         - school_locations_RM.parquet          ← una fila por RBD con lat/lon canónico

Notas:
    - Coordenada madre usa MODA (no mediana) sobre coordenadas redondeadas a 4
      decimales (~11m). Garantiza que la ubicación final es un valor que existió
      realmente en el registro administrativo. Robusta ante outliers de captura.
    - La imputación de 2013 es desde el 2014: si el RBD no aparece en
      2014, queda con NaN y será excluido de los buffers espaciales.

Inputs:
    data/raw/Directorios_mineduc/Directorio_oficial_EE_YYYY.csv  (2013–2019)

Outputs:
    data/processed/schools_clean_2013_2019_RM.parquet
    data/processed/school_locations_RM.parquet

Ejecución:
    python scripts/04_dataprep_establecimientos.py
"""

import logging
import sys

import numpy as np
import pandas as pd

from project_paths import (
    DIRECTORIOS_MINEDUC_DIR,
    SCHOOL_LOCATIONS_PATH,
    SCHOOLS_OUTPUT_PATH,
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
YEARS          = list(range(2013, 2020))
COORD_DECIMALS = 4       # ~11m de precisión
OUTLIER_KM     = 1.0     # umbral para reportar coord outlier
COD_RM         = 13      # código Región Metropolitana

LAT_MIN, LAT_MAX = -56.0, -17.0
LON_MIN, LON_MAX = -76.0, -66.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine(lat1: pd.Series, lon1: pd.Series,
              lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Distancia haversine en metros (vectorizado)."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def guardar_parquet(df: pd.DataFrame, path, descripcion: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Guardado: %s — %d filas × %d cols  [%s]",
             path.name, len(df), len(df.columns), descripcion)


# ---------------------------------------------------------------------------
# Paso 1 — Carga de directorios anuales
# ---------------------------------------------------------------------------

def read_directorio_year(year: int) -> pd.DataFrame:
    """
    Carga un directorio MINEDUC anual de forma robusta.
    El año 2013 se carga normalmente pero sin lat/lon (se imputan en paso 3).
    """
    path = DIRECTORIOS_MINEDUC_DIR / f"Directorio_oficial_EE_{year}.csv"
    if not path.exists():
        log.error("Archivo no encontrado: %s", path)
        sys.exit(1)

    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    # Arregla BOM residual en columna agno (ïagno o variantes)
    bom_cols = [c for c in df.columns if "agno" in c and c != "agno"]
    if bom_cols:
        df = df.rename(columns={bom_cols[0]: "agno"})

    rename_map = {
        "rbd": "rbd", "agno": "agno",
        "latitud": "latitud", "longitud": "longitud",
        "cod_reg_rbd": "cod_reg_rbd", "cod_pro_rbd": "cod_pro_rbd",
        "cod_com_rbd": "cod_com_rbd", "nom_com_rbd": "nom_com_rbd",
        "cod_depe": "cod_depe", "cod_depe2": "cod_depe2",
        "rural_rbd": "rural_rbd", "pago_matricula": "pago_matricula",
        "pago_mensual": "pago_mensual", "ori_religiosa": "ori_religiosa",
        "estado_estab": "estado_estab",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "agno" not in df.columns:
        df["agno"] = year

    # 2013 no trae lat/lon — los forzamos a NaN para que el panel sea uniforme
    if year == 2013:
        df["latitud"]  = np.nan
        df["longitud"] = np.nan

    for col in ["rbd", "agno", "cod_reg_rbd", "cod_pro_rbd", "cod_com_rbd",
                "cod_depe", "cod_depe2", "rural_rbd", "ori_religiosa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["latitud", "longitud"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["nom_com_rbd", "pago_matricula", "pago_mensual", "estado_estab"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    tiene_coords = df["latitud"].notna().any() if "latitud" in df.columns else False
    log.info("  %d — %d filas | lat/lon: %s",
             year, len(df), "sí" if tiene_coords else "NO (se imputa desde 2014)")
    return df


def cargar_panel() -> pd.DataFrame:
    log.info("Cargando directorios MINEDUC %d–%d...", YEARS[0], YEARS[-1])
    dfs = [read_directorio_year(y) for y in YEARS]
    raw = pd.concat(dfs, ignore_index=True)
    log.info("Panel crudo: %d filas", len(raw))
    return raw


# ---------------------------------------------------------------------------
# Paso 2 — Selección de columnas y limpieza básica
# ---------------------------------------------------------------------------

def limpiar_panel(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "rbd", "agno", "latitud", "longitud",
        "cod_reg_rbd", "cod_pro_rbd", "cod_com_rbd", "nom_com_rbd",
        "cod_depe", "cod_depe2", "rural_rbd",
        "pago_matricula", "pago_mensual", "ori_religiosa", "estado_estab",
    ]
    for i in range(1, 10):
        c = f"ens_{i:02d}"
        if c in df.columns:
            keep.append(c)

    keep    = [c for c in keep if c in df.columns]
    schools = df[keep].copy()

    schools["rbd"]  = pd.to_numeric(schools["rbd"],  errors="coerce").astype("Int64")
    schools["agno"] = pd.to_numeric(schools["agno"], errors="coerce").astype("Int64")
    schools = schools.dropna(subset=["rbd", "agno"])

    dups = schools.duplicated(subset=["rbd", "agno"]).sum()
    if dups:
        log.warning("Duplicados rbd-agno detectados: %d — se eliminan", dups)
        schools = schools.drop_duplicates(subset=["rbd", "agno"])

    log.info("Panel limpio: %d filas × %d cols", len(schools), len(schools.columns))
    return schools


# ---------------------------------------------------------------------------
# Paso 3 — Imputar lat/lon de 2013 desde 2014
# ---------------------------------------------------------------------------

def imputar_coords_2013(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copia latitud/longitud de 2014 a los registros de 2013 del mismo RBD.
    Se hace antes de calcular la coordenada madre para que 2013 participe
    correctamente en la moda.
    """
    coords_2014 = (
        df[df["agno"] == 2014][["rbd", "latitud", "longitud"]]
        .dropna(subset=["latitud", "longitud"])
        .drop_duplicates(subset="rbd")
        .rename(columns={"latitud": "lat_imp", "longitud": "lon_imp"})
    )

    mask_2013 = df["agno"] == 2013
    df = df.merge(coords_2014, on="rbd", how="left")
    df.loc[mask_2013, "latitud"]  = df.loc[mask_2013, "lat_imp"]
    df.loc[mask_2013, "longitud"] = df.loc[mask_2013, "lon_imp"]
    df = df.drop(columns=["lat_imp", "lon_imp"])

    n_2013 = int(mask_2013.sum())
    n_imp  = int(df[mask_2013].dropna(subset=["latitud", "longitud"]).shape[0])
    n_sin  = int(df[mask_2013 & df["latitud"].isna()]["rbd"].nunique())

    log.info("Imputación 2013 desde 2014: %d/%d RBDs con coords (%.1f%%)",
             n_imp, n_2013, 100 * n_imp / n_2013)
    if n_sin:
        log.warning("2013 — %d RBDs sin coordenadas (no aparecen en 2014)", n_sin)
    return df


# ---------------------------------------------------------------------------
# Paso 4 — Validación de coordenadas
# ---------------------------------------------------------------------------

def validar_coordenadas(df: pd.DataFrame) -> None:
    total       = len(df)
    with_coords = df.dropna(subset=["latitud", "longitud"]).shape[0]
    log.info("Cobertura de coordenadas: %d/%d (%.1f%%)",
             with_coords, total, 100 * with_coords / total)

    bad = df.dropna(subset=["latitud", "longitud"]).query(
        "(latitud < @LAT_MIN) or (latitud > @LAT_MAX) "
        "or (longitud < @LON_MIN) or (longitud > @LON_MAX)"
    )
    if len(bad):
        log.warning("Coordenadas fuera del rango plausible Chile: %d filas", len(bad))
    else:
        log.info("Rango de coordenadas OK ✓")

    # Estabilidad año a año (solo 2014–2019; 2013 fue imputado desde 2014)
    for y1, y2 in [(y, y + 1) for y in range(2014, 2019)]:
        a = df.query("agno == @y1")[["rbd", "latitud", "longitud"]]
        b = df.query("agno == @y2")[["rbd", "latitud", "longitud"]]
        merged = a.merge(b, on="rbd", suffixes=(f"_{y1}", f"_{y2}")).dropna()
        if merged.empty:
            continue
        merged["dist_m"] = haversine(
            merged[f"latitud_{y1}"], merged[f"longitud_{y1}"],
            merged[f"latitud_{y2}"], merged[f"longitud_{y2}"]
        )
        log.info(
            "  %d→%d | n=%d | mediana=%.0fm | p95=%.0fm | >500m: %d | >1km: %d",
            y1, y2, len(merged),
            merged["dist_m"].median(),
            merged["dist_m"].quantile(0.95),
            (merged["dist_m"] > 500).sum(),
            (merged["dist_m"] > 1000).sum(),
        )


# ---------------------------------------------------------------------------
# Paso 5 — Coordenada madre por RBD
# ---------------------------------------------------------------------------

def calcular_coordenada_madre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la coordenada madre usando la moda sobre coordenadas redondeadas
    a 4 decimales (~11m). En caso de empate, se toma el primer valor (más antiguo).
    """
    coord_madre = (
        df.dropna(subset=["latitud", "longitud"])
        .assign(
            latitud  = lambda d: d["latitud"].round(COORD_DECIMALS),
            longitud = lambda d: d["longitud"].round(COORD_DECIMALS),
        )
        .groupby("rbd")[["latitud", "longitud"]]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"latitud": "lat_madre", "longitud": "lon_madre"})
    )
    log.info("Coordenada madre calculada para %d RBDs únicos", len(coord_madre))

    check = (
        df.merge(coord_madre, on="rbd", how="left")
        .dropna(subset=["latitud", "longitud", "lat_madre", "lon_madre"])
        .assign(dist_madre=lambda d: haversine(
            d["latitud"], d["longitud"], d["lat_madre"], d["lon_madre"]
        ))
    )
    outliers = check[check["dist_madre"] > OUTLIER_KM * 1000]
    if len(outliers):
        log.warning("Registros con coord anual >%.0fkm de la madre: %d (reemplazados)",
                    OUTLIER_KM, len(outliers))
        log.info("Detalle (rbd, agno, dist_m):\n%s",
                 outliers[["rbd", "agno", "dist_madre"]]
                 .sort_values("dist_madre", ascending=False)
                 .head(10).to_string(index=False))
    else:
        log.info("Sin outliers de coordenadas ✓")

    return coord_madre


# ---------------------------------------------------------------------------
# Paso 6 — Aplicar coordenada madre al panel
# ---------------------------------------------------------------------------

def aplicar_coordenada_madre(df: pd.DataFrame,
                              coord_madre: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["latitud", "longitud"], errors="ignore")
    df = df.merge(coord_madre, on="rbd", how="left")
    df = df.rename(columns={"lat_madre": "latitud", "lon_madre": "longitud"})

    sin_coords = df["latitud"].isna().sum()
    if sin_coords:
        log.warning("%d filas sin coordenada madre (RBDs sin ningún año con lat/lon)",
                    sin_coords)
    else:
        log.info("Coordenada madre aplicada a todas las filas ✓")
    return df


# ---------------------------------------------------------------------------
# Paso 7 — Filtrar Región Metropolitana
# ---------------------------------------------------------------------------

def filtrar_rm(df: pd.DataFrame) -> pd.DataFrame:
    antes  = len(df)
    df_rm  = df[df["cod_reg_rbd"] == COD_RM].copy()
    log.info("Filtro RM (cod_reg_rbd==%d): %d → %d filas | RBDs únicos: %d",
             COD_RM, antes, len(df_rm), df_rm["rbd"].nunique())
    return df_rm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Script 04: Preparación de establecimientos 2013–2019 RM ===")

    # 1 — Cargar todos los años
    schools_raw = cargar_panel()

    # 2 — Limpiar y seleccionar columnas
    schools = limpiar_panel(schools_raw)

    # 3 — Imputar coords de 2013 desde 2014
    schools = imputar_coords_2013(schools)

    # 4 — Validar coordenadas y estabilidad
    validar_coordenadas(schools)

    # 5 — Calcular coordenada madre
    coord_madre = calcular_coordenada_madre(schools)

    # 6 — Aplicar coordenada madre
    schools = aplicar_coordenada_madre(schools, coord_madre)

    # 7 — Filtrar Región Metropolitana
    schools_rm = filtrar_rm(schools)

    # 8 — Tabla de ubicaciones: una fila por RBD (input del Script 07)
    locations_rm = (
        coord_madre
        .merge(
            schools_rm[["rbd", "cod_reg_rbd", "cod_com_rbd", "nom_com_rbd", "cod_depe2"]]
            .drop_duplicates(subset="rbd"),
            on="rbd", how="inner"
        )
        .rename(columns={"lat_madre": "latitud", "lon_madre": "longitud"})
    )

    # 9 — Guardar
    guardar_parquet(schools_rm,   SCHOOLS_OUTPUT_PATH,   "panel 2013–2019 RM con coord madre")
    guardar_parquet(locations_rm, SCHOOL_LOCATIONS_PATH, "una fila por RBD con lat/lon canónico")

    log.info("── Resumen final ──")
    log.info("  Filas:       %d", len(schools_rm))
    log.info("  RBDs únicos: %d", schools_rm["rbd"].nunique())
    log.info("  Años:        %s", sorted(schools_rm["agno"].dropna().unique().tolist()))
    log.info("  Con coords:  %d (%.1f%%)",
             schools_rm.dropna(subset=["latitud", "longitud"]).shape[0],
             100 * schools_rm.dropna(subset=["latitud", "longitud"]).shape[0] / len(schools_rm))
    log.info("  Locations:   %d RBDs únicos", len(locations_rm))


if __name__ == "__main__":
    main()
