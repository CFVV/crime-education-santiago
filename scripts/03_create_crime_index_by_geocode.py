"""
03_create_crime_index_by_geocode.py
====================================
Construye el índice de criminalidad por geocódigo y fecha.

Outputs (en data/intermediate/):
┌─────────────────────────────────────────┬──────────────────────────────────────┐
│ Archivo                                 │ Granularidad                         │
├─────────────────────────────────────────┼──────────────────────────────────────┤
│ crime_index_geocode_daily.csv           │ GEOCODIGO + fecha diaria — agregado  │
│ crime_index_geocode_monthly.csv         │ GEOCODIGO + mes          — agregado  │
│ crime_detail_geocode_daily.csv          │ GEOCODIGO + fecha + delito           │
│ crime_detail_geocode_monthly.csv        │ GEOCODIGO + mes   + delito           │
└─────────────────────────────────────────┴──────────────────────────────────────┘

Columnas archivos AGREGADOS:
    GEOCODIGO, fecha/mes, anio
    indice_delictualidad    → promedio ponderado de severidad [0, 1]
    frecuencia_total        → total de delitos en ese geocode+fecha
    n_delitos_distintos     → cuántos tipos de delito distintos
    n_nivel_1 … n_nivel_8   → frecuencia por nivel de gravedad

Columnas archivos DETALLADOS:
    GEOCODIGO, fecha/mes, anio
    delito, nivel_gravedad, grupo_gravedad, rank_gravedad, score_gravedad
    frecuencia              → frecuencia del delito específico ese día/mes
    indice_delictualidad    → índice agregado del geocode+fecha (repetido por fila)

Fórmula del índice:
    weighted_score       = frecuencia * score_gravedad
    indice_delictualidad = sum(weighted_score) / sum(frecuencia)
    → 0 si no hay delitos

Ejecución:
    python scripts/03_create_crime_index_by_geocode.py
"""

import logging
import sys

import pandas as pd

from project_paths import (
    CRIME_DETAIL_DAILY_PATH,
    CRIME_DETAIL_MONTHLY_PATH,
    CRIME_INDEX_DAILY_PATH,
    CRIME_INDEX_MONTHLY_PATH,
    CRIME_SEVERITY_LOOKUP_PATH,
    INTERMEDIATE_DATA_DIR,
    RAW_CRIME_DATA_PATH,
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
# Constantes
# ---------------------------------------------------------------------------
NIVELES = list(range(1, 9))   # niveles 1 a 8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def leer_csv_con_encoding(path, **kwargs) -> pd.DataFrame:
    """Intenta UTF-8 primero; si falla, cae a latin-1."""
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
            log.info("  '%s' leído con encoding '%s' — %d filas", path.name, enc, len(df))
            return df
        except UnicodeDecodeError:
            log.warning("  Encoding '%s' falló para '%s', intentando siguiente...", enc, path.name)
    raise ValueError(f"No se pudo leer {path} con utf-8 ni latin-1.")


def guardar_csv(df: pd.DataFrame, path, descripcion: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Guardado: %s — %d filas × %d columnas  [%s]",
             path.name, len(df), len(df.columns), descripcion)


# ---------------------------------------------------------------------------
# Carga y merge
# ---------------------------------------------------------------------------

def cargar_datos() -> pd.DataFrame:
    """Lee el CSV de delitos y el lookup, los valida y los une."""
    log.info("── Cargando datos ──")

    # Verificar existencia
    for p in (RAW_CRIME_DATA_PATH, CRIME_SEVERITY_LOOKUP_PATH):
        if not p.exists():
            log.error("Archivo no encontrado: %s", p)
            sys.exit(1)

    df_raw    = leer_csv_con_encoding(RAW_CRIME_DATA_PATH)
    df_lookup = leer_csv_con_encoding(CRIME_SEVERITY_LOOKUP_PATH)

    # Validar columnas requeridas
    required_raw    = {"fecha", "GEOCODIGO", "delito", "frecuencia"}
    required_lookup = {"delito", "rank_gravedad", "score_gravedad",
                       "nivel_gravedad", "grupo_gravedad"}

    for req, df, nombre in [
        (required_raw,    df_raw,    "listado_delitos.csv"),
        (required_lookup, df_lookup, "crime_severity_lookup.csv"),
    ]:
        faltantes = req - set(df.columns)
        if faltantes:
            raise ValueError(f"Faltan columnas en {nombre}: {sorted(faltantes)}")

    # Limpiar strings
    df_raw["delito"]    = df_raw["delito"].astype(str).str.strip()
    df_lookup["delito"] = df_lookup["delito"].astype(str).str.strip()

    # Merge
    n_antes = len(df_raw)
    df = df_raw.merge(
        df_lookup[["delito", "rank_gravedad", "score_gravedad",
                   "nivel_gravedad", "grupo_gravedad"]],
        on="delito",
        how="left",
    )

    # Advertir si quedaron delitos sin score (no deberían si Script 01 corrió bien)
    sin_score = df["score_gravedad"].isna().sum()
    if sin_score > 0:
        delitos_sin_score = df.loc[df["score_gravedad"].isna(), "delito"].unique()
        log.warning("%d filas sin score_gravedad (delitos no encontrados en lookup):", sin_score)
        for d in sorted(delitos_sin_score):
            log.warning("  - '%s'", d)
        log.warning("Estas filas serán excluidas del índice.")
        df = df.dropna(subset=["score_gravedad"])

    log.info("Filas tras merge: %d (de %d originales)", len(df), n_antes)

    # Parsear fecha y extraer componentes temporales
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    n_fecha_nula = df["fecha"].isna().sum()
    if n_fecha_nula > 0:
        log.warning("%d filas con fecha no parseable — serán excluidas", n_fecha_nula)
        df = df.dropna(subset=["fecha"])

    df["anio"] = df["fecha"].dt.year
    df["mes"]  = df["fecha"].dt.to_period("M").astype(str)   # "2013-01"
    df["dia"]  = df["fecha"].dt.date.astype(str)             # "2013-01-02"

    # Asegurar tipos correctos
    df["frecuencia"]     = pd.to_numeric(df["frecuencia"],     errors="coerce").fillna(0).astype(int)
    df["score_gravedad"] = pd.to_numeric(df["score_gravedad"], errors="coerce")
    df["GEOCODIGO"]      = df["GEOCODIGO"].astype(str).str.strip()

    # Columna de puntaje ponderado (usada en todas las agregaciones)
    df["weighted_score"] = df["frecuencia"] * df["score_gravedad"]

    log.info("Rango temporal: %s → %s", df["fecha"].min().date(), df["fecha"].max().date())
    log.info("GEOCODIGOs únicos: %d", df["GEOCODIGO"].nunique())
    log.info("Delitos distintos: %d", df["delito"].nunique())

    return df


# ---------------------------------------------------------------------------
# Construcción del índice agregado
# ---------------------------------------------------------------------------

def construir_indice(df: pd.DataFrame, grupo_temporal: list[str]) -> pd.DataFrame:
    """
    Agrega el índice de criminalidad por geocode + dimensión temporal.

    grupo_temporal: lista de columnas temporales, ej. ["dia", "anio"] o ["mes", "anio"]
    """
    group_cols = ["GEOCODIGO"] + grupo_temporal

    # Índice principal
    agg = df.groupby(group_cols, sort=False).agg(
        indice_delictualidad=("weighted_score",  lambda x: (
            x.sum() / df.loc[x.index, "frecuencia"].sum()
            if df.loc[x.index, "frecuencia"].sum() > 0 else 0.0
        )),
        frecuencia_total    =("frecuencia",      "sum"),
        n_delitos_distintos =("delito",          "nunique"),
    ).reset_index()

    # Conteo de frecuencia por nivel de gravedad (columnas n_nivel_1 … n_nivel_8)
    for nivel in NIVELES:
        mask = df["nivel_gravedad"] == nivel
        nivel_counts = (
            df[mask]
            .groupby(group_cols, sort=False)["frecuencia"]
            .sum()
            .reset_index()
            .rename(columns={"frecuencia": f"n_nivel_{nivel}"})
        )
        agg = agg.merge(nivel_counts, on=group_cols, how="left")
        agg[f"n_nivel_{nivel}"] = agg[f"n_nivel_{nivel}"].fillna(0).astype(int)

    # Redondear índice
    agg["indice_delictualidad"] = agg["indice_delictualidad"].round(6)

    return agg


# ---------------------------------------------------------------------------
# Construcción del detalle por delito
# ---------------------------------------------------------------------------

def construir_detalle(df: pd.DataFrame,
                      grupo_temporal: list[str],
                      df_index: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega por geocode + dimensión temporal + delito.
    Adjunta el índice agregado para referencia cruzada.
    """
    group_cols = ["GEOCODIGO"] + grupo_temporal + [
        "delito", "nivel_gravedad", "grupo_gravedad",
        "rank_gravedad", "score_gravedad",
    ]

    detalle = df.groupby(group_cols, sort=False).agg(
        frecuencia=("frecuencia", "sum"),
    ).reset_index()

    # Pegar índice agregado
    merge_cols = ["GEOCODIGO"] + grupo_temporal
    detalle = detalle.merge(
        df_index[merge_cols + ["indice_delictualidad"]],
        on=merge_cols,
        how="left",
    )

    # Ordenar
    detalle = detalle.sort_values(
        ["GEOCODIGO"] + grupo_temporal + ["rank_gravedad"]
    ).reset_index(drop=True)

    return detalle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Script 03: Crear índice de criminalidad por geocódigo ===")

    # 1. Cargar y unir datos
    df = cargar_datos()

    # 2. Índices agregados
    log.info("── Construyendo índices agregados ──")

    log.info("Procesando nivel diario...")
    idx_daily   = construir_indice(df, grupo_temporal=["dia", "anio"])

    log.info("Procesando nivel mensual...")
    idx_monthly = construir_indice(df, grupo_temporal=["mes", "anio"])

    # 3. Detalles por delito
    log.info("── Construyendo tablas detalladas ──")

    log.info("Procesando detalle diario...")
    det_daily   = construir_detalle(df, ["dia",  "anio"], idx_daily)

    log.info("Procesando detalle mensual...")
    det_monthly = construir_detalle(df, ["mes",  "anio"], idx_monthly)

    # 4. Guardar
    log.info("── Guardando archivos ──")
    INTERMEDIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    guardar_csv(idx_daily,   CRIME_INDEX_DAILY_PATH,   "índice diario agregado")
    guardar_csv(idx_monthly, CRIME_INDEX_MONTHLY_PATH, "índice mensual agregado")
    guardar_csv(det_daily,   CRIME_DETAIL_DAILY_PATH,  "detalle diario")
    guardar_csv(det_monthly, CRIME_DETAIL_MONTHLY_PATH,"detalle mensual")

    # 5. Resumen final
    log.info("── Resumen ──")
    log.info("Índice diario:    %d filas (geocodes × días)",   len(idx_daily))
    log.info("Índice mensual:   %d filas (geocodes × meses)",  len(idx_monthly))
    log.info("Detalle diario:   %d filas (geocode × día × delito)",   len(det_daily))
    log.info("Detalle mensual:  %d filas (geocode × mes × delito)",   len(det_monthly))
    log.info("Columnas índice:  %s", list(idx_daily.columns))
    log.info("Columnas detalle: %s", list(det_daily.columns))


if __name__ == "__main__":
    main()
