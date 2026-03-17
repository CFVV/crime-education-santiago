"""
05_dataprep_education.py
=========================
Construye la tabla final de educación: data_education_2013_2019_RM.parquet

Tareas:
    1. Carga simce_completo_5.dta (2013–2019, rbd-agno-grado, 211 cols)
    2. Filtra años 2013–2019
    3. Selecciona solo columnas relevantes:
         - llaves:    rbd, agno, grado
         - admin:     cod_reg_rbd, nom_reg_rbd, cod_pro_rbd, cod_com_rbd,
                      nom_com_rbd, cod_depe, cod_depe2, rural_rbd, nom_rbd
         - outcomes:  columnas con puntajes SIMCE (punt_*, prom_*)
         - EXCLUYE:   columnas de crimen del archivo original
                      (serán reemplazadas por el índice nuevo)
    4. Filtra Región Metropolitana (cod_reg_rbd == 13)
    5. Merge con schools_clean_2013_2019_RM.parquet
         → agrega latitud/longitud (coordenada madre)
    6. Guarda data_education_2013_2019_RM.parquet

Ejecución:
    python scripts/05_dataprep_education.py
"""

import logging
import sys

import pandas as pd

from project_paths import (
    EDUCATION_OUTPUT_PATH,
    SCHOOLS_OUTPUT_PATH,
    SIMCE_RAW_PATH,
)

# Alias para mantener nombres internos consistentes
SIMCE_PATH   = SIMCE_RAW_PATH
SCHOOLS_PATH = SCHOOLS_OUTPUT_PATH
OUTPUT_PATH  = EDUCATION_OUTPUT_PATH

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

COD_RM          = 13
ANIO_MIN        = 2013
ANIO_MAX        = 2019

# ---------------------------------------------------------------------------
# Columnas
# ---------------------------------------------------------------------------

# Columnas de crimen del archivo original — EXCLUIR
# (serán reemplazadas por el índice construido en Script 03)
CRIME_PATTERNS = [
    "homicidio", "secuestro", "violacion", "trata", "trafico_arma",
    "robo", "lesion", "violencia_intrafamiliar", "bomba", "molotov",
    "trafico_droga", "microtrafico", "cultivo", "posesion", "porte",
    "receptacion", "hurto", "abigeato", "incendio", "abuso_sexual",
    "delito",                        # captura genérica: delito_*, delitos_*
    "propiedad_intelectual",
    "economico", "falsificacion", "soborno",
    "exacciones", "trafico_emigrante", "cuasidelito", "danio", "danyo",
    "dano", "hallazgo", "desacato", "desorden", "salud_publica",
    "infraccion", "consumo", "ebriedad", "objetos_robados",
    "bombas_mo", "trafico_de_armas", "exacciones_legales",
    "indice_delictualidad", "cluster", "weighted", "score_gravedad",
]

# Columnas administrativas a conservar
ADMIN_COLS = [
    "rbd", "agno", "grado",
    "codigo_bbdd",
    "nom_rbd", "dvrbd",
    "cod_reg_rbd", "nom_reg_rbd",
    "cod_pro_rbd", "nom_pro_rbd",
    "cod_com_rbd", "nom_com_rbd",
    "cod_depe", "cod_depe2",
    "rural_rbd",
    "fecha_rendicion",   # fecha exacta del examen SIMCE ← crítico para Script 06
    "id_rbd_grado",      # identificador rbd-grado ← útil para joins
]

# Patrones de columnas de outcomes SIMCE a conservar
OUTCOME_PATTERNS = [
    "punt_", "prom_", "puntaje", "promedio",
    "lect", "leng", "lenguaje",
    "mat", "matematica",
    "cod_grupo",   # grupo socioeconómico — control clave en regresiones
    "n_alumnos", "nalu",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def es_columna_crimen(col: str) -> bool:
    col_lower = col.lower()
    return any(p in col_lower for p in CRIME_PATTERNS)


def es_columna_outcome(col: str) -> bool:
    col_lower = col.lower()
    return any(p in col_lower for p in OUTCOME_PATTERNS)


def seleccionar_columnas(df: pd.DataFrame) -> list[str]:
    """
    Selecciona columnas relevantes:
        - Admin cols presentes en el df
        - Outcome cols detectadas por patrón
        - Excluye explícitamente columnas de crimen
    """
    todas = df.columns.tolist()

    # Admin
    admin = [c for c in ADMIN_COLS if c in todas]

    # Outcomes: detectados por patrón Y no son crimen
    outcomes = [
        c for c in todas
        if es_columna_outcome(c)
        and not es_columna_crimen(c)
        and c not in admin
    ]

    # Crimen: para log
    crimen_excluido = [c for c in todas if es_columna_crimen(c)]

    log.info("Columnas admin seleccionadas:    %d", len(admin))
    log.info("Columnas outcome seleccionadas:  %d", len(outcomes))
    log.info("Columnas crimen excluidas:       %d → %s",
             len(crimen_excluido), crimen_excluido[:10])

    otras_excluidas = [
        c for c in todas
        if c not in admin
        and c not in outcomes
        and c not in crimen_excluido
    ]
    log.info("Otras columnas excluidas:        %d → %s",
             len(otras_excluidas), otras_excluidas[:10])

    return admin + outcomes


def guardar_parquet(df: pd.DataFrame, path, descripcion: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Guardado: %s — %d filas × %d cols  [%s]",
             path.name, len(df), len(df.columns), descripcion)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Script 05: Preparación final de data_education ===")

    # 1. Verificar archivos
    for p in (SIMCE_PATH, SCHOOLS_PATH):
        if not p.exists():
            log.error("No se encontró: %s", p)
            sys.exit(1)

    # 2. Cargar SIMCE
    log.info("Cargando %s ...", SIMCE_PATH.name)
    simce = pd.read_stata(SIMCE_PATH, convert_categoricals=False)
    simce.columns = [c.strip().lower() for c in simce.columns]
    log.info("SIMCE cargado: %d filas × %d cols", len(simce), len(simce.columns))

    # 3. Tipado de llaves
    for col in ["rbd", "agno", "cod_reg_rbd", "cod_depe", "rural_rbd"]:
        if col in simce.columns:
            simce[col] = pd.to_numeric(simce[col], errors="coerce")

    simce["rbd"]  = simce["rbd"].astype("Int64")
    simce["agno"] = simce["agno"].astype("Int64")

    if "grado" in simce.columns:
        simce["grado"] = simce["grado"].astype(str).str.strip().str.lower()

    # cod_depe2: 2013-2017 usan 1=Municipal,2=Part.Sub,3=Part.Pag | 2018-2019 usan 4,5,6 para lo mismo
    if "cod_depe2" in simce.columns:
        DEPE2_MAP = {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}  # 7 = residual → NaN
        simce["cod_depe2"] = (pd.to_numeric(simce["cod_depe2"], errors="coerce")
                               .map(DEPE2_MAP))
        log.info("cod_depe2 — no-nulos: %d | distribución: %s",
                 simce["cod_depe2"].notna().sum(),
                 simce["cod_depe2"].value_counts(dropna=False).sort_index().to_dict())

    # cod_grupo: 2013-2017 usan 1-5 | 2018-2019 usan 6-10 para lo mismo
    if "cod_grupo" in simce.columns:
        GRUPO_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
                     6: 1, 7: 2, 8: 3, 9: 4, 10: 5}
        simce["cod_grupo"] = (pd.to_numeric(simce["cod_grupo"], errors="coerce")
                               .map(GRUPO_MAP))
        log.info("cod_grupo — NaN: %d | distribución: %s",
                 simce["cod_grupo"].isna().sum(),
                 simce["cod_grupo"].value_counts(dropna=False).sort_index().to_dict())
    # 4. Filtrar período
    simce = simce.query("agno >= @ANIO_MIN and agno <= @ANIO_MAX").copy()
    log.info("Tras filtro 2013–2019: %d filas | años: %s",
             len(simce), sorted(simce["agno"].dropna().unique().tolist()))

    # 5. Filtrar RM
    simce_rm = simce[simce["cod_reg_rbd"] == COD_RM].copy()
    log.info("Tras filtro RM: %d filas | RBDs únicos: %d",
             len(simce_rm), simce_rm["rbd"].nunique())

    # 6. Seleccionar columnas
    keep = seleccionar_columnas(simce_rm)
    simce_rm = simce_rm[keep].copy()
    log.info("Columnas finales seleccionadas: %d", len(simce_rm.columns))

    # 7. Verificar duplicados por llave rbd-agno-grado
    key = ["rbd", "agno", "grado"]
    dups = simce_rm.duplicated(subset=key).sum()
    if dups > 0:
        log.warning("Duplicados rbd-agno-grado: %d — se eliminan (keep=first)", dups)
        simce_rm = simce_rm.drop_duplicates(subset=key, keep="first")

    # 8. Merge con coordenadas (schools_clean_2013_2019_RM)
    log.info("Cargando %s ...", SCHOOLS_PATH.name)
    schools = pd.read_parquet(SCHOOLS_PATH)
    schools.columns = [c.strip().lower() for c in schools.columns]
    schools["rbd"]  = pd.to_numeric(schools["rbd"],  errors="coerce").astype("Int64")
    schools["agno"] = pd.to_numeric(schools["agno"], errors="coerce").astype("Int64")

    # Solo traemos lat/lon del schools (lo demás ya viene del SIMCE)
    coords = schools[["rbd", "agno", "latitud", "longitud"]].drop_duplicates(
        subset=["rbd", "agno"]
    )

    data_edu = simce_rm.merge(coords, on=["rbd", "agno"], how="left")

    # 9. Reporte de cobertura de coordenadas
    total_rbd  = data_edu["rbd"].nunique()
    coords_ok  = data_edu.dropna(subset=["latitud", "longitud"])["rbd"].nunique()
    filas_ok   = data_edu.dropna(subset=["latitud", "longitud"]).shape[0]

    log.info("Cobertura coordenadas:")
    log.info("  RBDs con coords:  %d / %d (%.1f%%)",
             coords_ok, total_rbd, 100 * coords_ok / total_rbd)
    log.info("  Filas con coords: %d / %d (%.1f%%)",
             filas_ok, len(data_edu), 100 * filas_ok / len(data_edu))

    if total_rbd - coords_ok > 0:
        sin_coords = (
            data_edu[data_edu["latitud"].isna()][["rbd", "agno"]]
            .drop_duplicates()
            .sort_values(["rbd", "agno"])
        )
        log.warning("RBDs sin coordenadas: %d (primeros 10):\n%s",
                    total_rbd - coords_ok,
                    sin_coords.head(10).to_string(index=False))

    # 10. Resumen por año y grado
    resumen = (
        data_edu.groupby(["agno", "grado"])
        .agg(
            n_escuelas=("rbd", "nunique"),
            con_coords=("latitud", lambda x: x.notna().sum()),
        )
        .reset_index()
        .sort_values(["agno", "grado"])
    )
    log.info("Distribución por año y grado:\n%s", resumen.to_string(index=False))

    # 11. Guardar
    guardar_parquet(data_edu, OUTPUT_PATH, "SIMCE 2013–2019 RM con coords")

    log.info("── Columnas finales ──")
    log.info("%s", list(data_edu.columns))


if __name__ == "__main__":
    main()
