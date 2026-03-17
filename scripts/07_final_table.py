"""
07_final_table.py
==================
Construye la tabla final uniendo:
    - data_education_2013_2019_RM   ← outcomes SIMCE + controles
    - crime_exposure_400m           ← índice exposición crimen buffer 400m
    - crime_exposure_1km            ← índice exposición crimen buffer 1km

Estructura output:
    Una fila por rbd × agno × grado × ventana_dias
    = 32,519 combos educación × 6 ventanas = ~195,114 filas

Columnas output:
    Llaves:       rbd, agno, grado, ventana_dias
    Identidad:    nom_rbd, cod_com_rbd, nom_com_rbd, cod_reg_rbd
    Admin:        cod_depe2, rural_rbd, cod_grupo
    Outcomes:     prom_lect*, prom_mate*, nalu_* (según grado)
    Ubicación:    latitud, longitud
    Crimen 400m:  indice_exp_400m, frecuencia_total_400m, n_manzanas_400m,
                  n_nivel_1_400m … n_nivel_8_400m
    Crimen 1km:   indice_exp_1km,  frecuencia_total_1km,  n_manzanas_1km,
                  n_nivel_1_1km  … n_nivel_8_1km
    Flags:        sin_cobertura_crimen, fecha_examen

Comunas sin cobertura (661 escuelas) — flag sin_cobertura_crimen=True:
    Pirque, San José de Maipo, Colina, Lampa, Tiltil, Buin,
    Calera de Tango, Paine, Melipilla, Alhué, Curacaví, María Pinto,
    San Pedro, Talagante, El Monte, Isla de Maipo, Peñaflor

    Estas escuelas se incluyen en la tabla pero deben EXCLUIRSE
    de los modelos de regresión (sus ceros en indice_exp no son
    ceros reales sino ausencia de datos).

Ejecución:
    python scripts/07_final_table.py
"""

import logging
import sys

import pandas as pd

from project_paths import (
    CRIME_BUFFER_400M_PATH,
    CRIME_BUFFER_1KM_PATH,
    EDUCATION_OUTPUT_PATH,
    FINAL_TABLE_PATH,
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

# Columnas de outcome SIMCE a conservar en la tabla final
# (puntajes promedio + n° alumnos evaluados, por grado)
OUTCOME_COLS = [
    # 2° básico
    "prom_lect2b_rbd", "nalu_lect2b_rbd",
    # 4° básico
    "prom_lect4b_rbd", "prom_mate4b_rbd",
    "nalu_lect4b_rbd", "nalu_mate4b_rbd",
    # 6° básico
    "prom_lect6b_rbd", "prom_mate6b_rbd",
    "nalu_lect6b_rbd", "nalu_mate6b_rbd",
    # 8° básico
    "prom_lect8b_rbd", "prom_mate8b_rbd",
    "nalu_lect8b_rbd", "nalu_mate8b_rbd",
    # 2° medio
    "prom_lect2m_rbd", "prom_mate2m_rbd",
    "nalu_lect2m_rbd", "nalu_mate2m_rbd",
]

# Columnas de control a conservar
CONTROL_COLS = [
    "cod_depe2",    # tipo de dependencia: Municipal / Part. Subvencionado / Part. Pagado
    "rural_rbd",    # 0=urbano, 1=rural
    "cod_grupo",    # grupo socioeconómico (1=Bajo … 5=Alto)
    "cod_com_rbd",  # código comuna
    "nom_com_rbd",  # nombre comuna
    "cod_reg_rbd",  # código región (todos = 13)
    "nom_rbd",      # nombre establecimiento
    "latitud",
    "longitud",
    "fecha_rendicion",
]

NIVELES = list(range(1, 9))

# Comunas sin cobertura en el CSV de crimen
# Comunas sin cobertura en el CSV de crimen — identificadas por cod_com_rbd
# (cod_com_rbd tiene formatos inconsistentes como texto, usamos los códigos
#  válidos de 5 dígitos >= 13000 que corresponden a RM)
CODIGOS_SIN_COBERTURA = {
    13202,  # Pirque
    13203,  # San José de Maipo
    13301,  # Colina
    13302,  # Lampa
    13303,  # Tiltil
    13402,  # Buin
    13403,  # Calera de Tango
    13404,  # Paine
    13501,  # Melipilla
    13502,  # Alhué
    13503,  # Curacaví
    13504,  # María Pinto
    13505,  # San Pedro
    13601,  # Talagante
    13602,  # El Monte
    13603,  # Isla de Maipo
    13604,  # Padre Hurtado
    13605,  # Peñaflor
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def guardar_parquet(df: pd.DataFrame, path, descripcion: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Guardado: %s — %d filas × %d cols  [%s]",
             path.name, len(df), len(df.columns), descripcion)


def renombrar_buffer(df: pd.DataFrame, sufijo: str) -> pd.DataFrame:
    """
    Renombra columnas del buffer agregando sufijo (_400m o _1km)
    para distinguirlas en la tabla final.
    Conserva las llaves sin cambio.
    """
    llaves = {"rbd", "agno", "grado", "ventana_dias", "fecha_examen"}
    return df.rename(columns={
        c: f"{c}_{sufijo}" for c in df.columns if c not in llaves
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Script 07: Tabla analítica final ===")

    # ------------------------------------------------------------------
    # 1. Cargar inputs
    # ------------------------------------------------------------------
    for p in (EDUCATION_OUTPUT_PATH, CRIME_BUFFER_400M_PATH, CRIME_BUFFER_1KM_PATH):
        if not p.exists():
            log.error("No se encontró: %s", p)
            sys.exit(1)

    log.info("Cargando data_education...")
    edu = pd.read_parquet(EDUCATION_OUTPUT_PATH)
    log.info("  %d filas × %d cols | RBDs: %d",
             len(edu), len(edu.columns), edu["rbd"].nunique())

    log.info("Cargando crime_exposure_400m...")
    buf400 = pd.read_parquet(CRIME_BUFFER_400M_PATH)
    log.info("  %d filas | ventanas: %s",
             len(buf400), sorted(buf400["ventana_dias"].unique().tolist()))

    log.info("Cargando crime_exposure_1km...")
    buf1km = pd.read_parquet(CRIME_BUFFER_1KM_PATH)
    log.info("  %d filas", len(buf1km))

    # ------------------------------------------------------------------
    # 2. Preparar educación
    #    Seleccionar columnas relevantes + normalizar tipos de llaves
    # ------------------------------------------------------------------
    key_edu   = ["rbd", "agno", "grado"]
    keep_cols = key_edu + [c for c in CONTROL_COLS + OUTCOME_COLS
                           if c in edu.columns]
    edu = edu[keep_cols].copy()

    edu["rbd"]  = edu["rbd"].astype("Int64")
    edu["agno"] = edu["agno"].astype(int)
    edu["grado"] = edu["grado"].astype(str).str.strip().str.lower()

    log.info("Educación preparada: %d filas × %d cols", len(edu), len(edu.columns))

    # ------------------------------------------------------------------
    # 3. Preparar buffers
    #    Normalizar tipos + renombrar columnas con sufijo
    # ------------------------------------------------------------------
    key_buf = ["rbd", "agno", "grado", "ventana_dias"]

    for buf in (buf400, buf1km):
        buf["rbd"]   = buf["rbd"].astype("Int64")
        buf["agno"]  = buf["agno"].astype(int)
        buf["grado"] = buf["grado"].astype(str).str.strip().str.lower()

    buf400 = renombrar_buffer(buf400, "400m")
    buf1km = renombrar_buffer(buf1km, "1km")

    # ------------------------------------------------------------------
    # 4. Expandir educación a nivel ventana_dias
    #    (cartesian con las ventanas, luego join con buffers)
    # ------------------------------------------------------------------
    ventanas   = sorted(buf400["ventana_dias"].unique().tolist())
    ventanas_df = pd.DataFrame({"ventana_dias": ventanas})

    log.info("Expandiendo educación × %d ventanas...", len(ventanas))
    edu_exp = edu.merge(ventanas_df, how="cross")
    log.info("  Filas tras expansión: %d", len(edu_exp))

    # ------------------------------------------------------------------
    # 5. Join con buffers
    # ------------------------------------------------------------------
    log.info("Join con buffer 400m...")
    tabla = edu_exp.merge(buf400, on=key_buf, how="left")
    log.info("  Filas tras join 400m: %d", len(tabla))

    # Recuperar fecha_examen desde buf400 (viene de fecha_rendicion, son iguales)
    # pero la renombramos a fecha_examen para claridad
    if "fecha_examen_400m" in tabla.columns:
        tabla["fecha_examen"] = tabla["fecha_examen_400m"]

    log.info("Join con buffer 1km...")
    tabla = tabla.merge(buf1km, on=key_buf, how="left")
    log.info("  Filas tras join 1km: %d", len(tabla))

    # Eliminar columnas fecha_examen duplicadas del buffer
    for c in ["fecha_examen_400m", "fecha_examen_1km"]:
        if c in tabla.columns:
            tabla = tabla.drop(columns=[c])

    # ------------------------------------------------------------------
    # 6. Flag sin_cobertura_crimen
    # ------------------------------------------------------------------
    log.info("Calculando flag sin_cobertura_crimen...")

    # cod_com_rbd tiene formatos inconsistentes según el año:
    # algunos años tienen códigos válidos de 5 dígitos (13202, 13301…)
    # otros tienen códigos truncados (1.0, 2.0…)
    # Estrategia: identificar los RBDs sin cobertura usando SOLO los registros
    # con códigos válidos (>= 13000), luego propagar el flag por rbd.

    cod_num = pd.to_numeric(tabla["cod_com_rbd"], errors="coerce")
    mask_valido = cod_num >= 13000

    rbd_sin_cobertura = set(
        tabla.loc[mask_valido & cod_num.isin(CODIGOS_SIN_COBERTURA), "rbd"]
        .unique()
        .tolist()
    )
    log.info("  RBDs sin cobertura identificados (via códigos válidos): %d",
             len(rbd_sin_cobertura))

    # Propagar a TODAS las filas del rbd, independiente del año/formato
    tabla["sin_cobertura_crimen"] = tabla["rbd"].isin(rbd_sin_cobertura)

    n_sin   = tabla[tabla["sin_cobertura_crimen"]]["rbd"].nunique()
    n_total = tabla["rbd"].nunique()
    log.info("  Escuelas sin cobertura (flag final): %d / %d (%.1f%%)",
             n_sin, n_total, 100 * n_sin / n_total)

    # Comunas detectadas (solo desde registros con código válido)
    comunas_detectadas = (
        tabla.loc[mask_valido & cod_num.isin(CODIGOS_SIN_COBERTURA),
                  ["cod_com_rbd", "nom_com_rbd"]]
        .drop_duplicates()
        .sort_values("cod_com_rbd")
    )
    log.info("  Comunas sin cobertura:\n%s",
             comunas_detectadas.drop_duplicates("cod_com_rbd").to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Columna de outcome "limpia" por grado
    #    Agrega prom_lect y prom_mate directamente según el grado de cada fila
    # ------------------------------------------------------------------
    log.info("Extrayendo puntaje del grado correspondiente...")

    def extraer_outcome(row, asignatura):
        col = f"prom_{asignatura}{row['grado']}_rbd"
        return row.get(col, pd.NA)

    tabla["prom_lect"] = tabla.apply(lambda r: extraer_outcome(r, "lect"), axis=1)
    tabla["prom_mate"] = tabla.apply(lambda r: extraer_outcome(r, "mate"), axis=1)
    tabla["nalu_lect"] = tabla.apply(
        lambda r: r.get(f"nalu_lect{r['grado']}_rbd", pd.NA), axis=1)

    log.info("  Cobertura prom_lect: %.1f%% no-nulo",
             100 * tabla["prom_lect"].notna().mean())
    log.info("  Cobertura prom_mate: %.1f%% no-nulo",
             100 * tabla["prom_mate"].notna().mean())

    # ------------------------------------------------------------------
    # 8. Ordenar columnas
    # ------------------------------------------------------------------
    cols_llaves   = ["rbd", "agno", "grado", "ventana_dias"]
    cols_id       = ["nom_rbd", "cod_com_rbd", "nom_com_rbd", "cod_reg_rbd"]
    cols_admin    = ["cod_depe2", "rural_rbd", "cod_grupo"]
    cols_geo      = ["latitud", "longitud"]
    cols_fecha    = ["fecha_rendicion", "fecha_examen"]
    cols_outcome  = ["prom_lect", "prom_mate", "nalu_lect"]
    cols_400m     = (["indice_exp_400m", "frecuencia_total_400m", "n_manzanas_400m"] +
                     [f"n_nivel_{n}_400m" for n in NIVELES])
    cols_1km      = (["indice_exp_1km",  "frecuencia_total_1km",  "n_manzanas_1km"] +
                     [f"n_nivel_{n}_1km"  for n in NIVELES])
    cols_flag     = ["sin_cobertura_crimen"]
    cols_outcome_raw = [c for c in tabla.columns
                        if c.startswith("prom_") and c.endswith("_rbd")]
    cols_nalu_raw    = [c for c in tabla.columns
                        if c.startswith("nalu_") and c.endswith("_rbd")]

    orden = (cols_llaves + cols_id + cols_admin + cols_geo + cols_fecha +
             cols_outcome + cols_400m + cols_1km + cols_flag +
             cols_outcome_raw + cols_nalu_raw)

    orden_final = [c for c in orden if c in tabla.columns]
    tabla = tabla[orden_final]

    # ------------------------------------------------------------------
    # 9. Resumen final
    # ------------------------------------------------------------------
    log.info("── Resumen tabla final ──")
    log.info("  Forma: %d filas × %d cols", len(tabla), len(tabla.columns))
    log.info("  RBDs únicos: %d", tabla["rbd"].nunique())
    log.info("  Ventanas: %s", sorted(tabla["ventana_dias"].unique().tolist()))

    resumen = (tabla.groupby(["agno", "grado"])
               .agg(
                   n_escuelas      = ("rbd", "nunique"),
                   pct_cero_400m   = ("indice_exp_400m",
                                      lambda x: round(100 * (x == 0).mean(), 1)),
                   mean_idx_400m   = ("indice_exp_400m",
                                      lambda x: round(x[x > 0].mean(), 4)),
                   pct_sin_cob     = ("sin_cobertura_crimen",
                                      lambda x: round(100 * x.mean(), 1)),
               )
               .reset_index()
               .sort_values(["agno", "grado"]))
    log.info("  Resumen por agno-grado (ventana_dias pooled):\n%s",
             resumen.to_string(index=False))

    # ------------------------------------------------------------------
    # 10. Guardar
    # ------------------------------------------------------------------
    guardar_parquet(tabla, FINAL_TABLE_PATH, "tabla analítica final")

    # También guardar versión CSV liviana (solo ventana=30d, sin columnas raw)
    tabla_30 = tabla[tabla["ventana_dias"] == 30].drop(
        columns=cols_outcome_raw + cols_nalu_raw, errors="ignore"
    )
    csv_path = FINAL_TABLE_PATH.parent / "tabla_final_30d.csv"
    tabla_30.to_csv(csv_path, index=False)
    log.info("CSV (ventana=30d): %s — %d filas × %d cols",
             csv_path.name, len(tabla_30), len(tabla_30.columns))

    log.info("=== Script 07 completado ===")


if __name__ == "__main__":
    main()
