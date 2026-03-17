"""
08_regression_analysis.py
==========================
Replica los modelos de regresión lineal del paper:
    "Impact of Crime Exposure on Educational Performance in Santiago de Chile"

Especificación por año × grado:
    prom_lect (o prom_mate) = β₀ + β₁·indice_exp + β₂·cod_grupo + β₃·cod_depe2 + ε

Controles:
    - cod_grupo:  grupo socioeconómico (Low como referencia)
    - cod_depe2:  tipo de dependencia (Municipal como referencia)

Dos índices de crimen:
    - indice_exp_400m  ← Global Crime Index (análogo a v2 del paper)
    - frecuencia_total_400m filtrada a niveles 1-3 ← Cluster Index (análogo a v3)

Ventana principal:    30 días (configurable via VENTANA_PRINCIPAL)
Radio principal:      400m   (configurable via RADIO_PRINCIPAL)

Outputs:
    - resultados/regression_global_index.csv     ← coeficientes por modelo
    - resultados/regression_cluster_index.csv
    - resultados/tabla_resumen_global.csv         ← tabla estilo paper (agno × grado)
    - resultados/tabla_resumen_cluster.csv

Ejecución:
    pip install statsmodels
    python scripts/08_regression_analysis.py

Dependencias:
    pip install statsmodels
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import FINAL_TABLE_PATH, PROCESSED_DATA_DIR

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
VENTANA_PRINCIPAL = 30     # días — ventana usada en el paper
RADIO_PRINCIPAL   = "400m" # radio — buffer principal del paper

RESULTADOS_DIR = PROCESSED_DATA_DIR.parent / "resultados"

# Columnas de outcome por grado (lectura y matemáticas)
OUTCOMES_POR_GRADO = {
    "2b": {"lect": "prom_lect", "mate": None},           # 2b solo tiene lectura
    "4b": {"lect": "prom_lect", "mate": "prom_mate"},
    "6b": {"lect": "prom_lect", "mate": "prom_mate"},
    "8b": {"lect": "prom_lect", "mate": "prom_mate"},
    "2m": {"lect": "prom_lect", "mate": "prom_mate"},
}

# Mapeos de categorías para dummies (Municipal y Bajo como referencia)
DEPE_REF    = 1   # Municipal (cod_depe2 = 1)
GRUPO_REF   = 1   # Bajo      (cod_grupo = 1)

DEPE_LABELS = {
    1: "Municipal",
    2: "Particular Subvencionado",
    3: "Particular Pagado",
}
GRUPO_LABELS = {
    1: "Bajo",
    2: "Medio Bajo",
    3: "Medio",
    4: "Medio Alto",
    5: "Alto",
}


# ---------------------------------------------------------------------------
# Preparar datos
# ---------------------------------------------------------------------------

def cargar_datos() -> pd.DataFrame:
    try:
        import statsmodels  # noqa
    except ImportError:
        log.error("Instala statsmodels: pip install statsmodels")
        sys.exit(1)

    if not FINAL_TABLE_PATH.exists():
        log.error("No se encontró: %s", FINAL_TABLE_PATH)
        log.error("Corre primero: python scripts/07_final_table.py")
        sys.exit(1)

    df = pd.read_parquet(FINAL_TABLE_PATH)
    log.info("Tabla final cargada: %d filas × %d cols", len(df), len(df.columns))

    # Filtrar universo de análisis
    df = df[
        (~df["sin_cobertura_crimen"]) &
        (df["ventana_dias"] == VENTANA_PRINCIPAL)
    ].copy()
    log.info("Tras filtros (sin_cobertura=False, ventana=%dd): %d filas | RBDs: %d",
             VENTANA_PRINCIPAL, len(df), df["rbd"].nunique())

    # Construir índice cluster: suma de crímenes niveles 1-3
    # (equivalente al Cluster-Based Crime Index v3 del paper)
    nivel_cols = [f"n_nivel_{n}_{RADIO_PRINCIPAL}" for n in [1, 2, 3]]
    cols_presentes = [c for c in nivel_cols if c in df.columns]
    if cols_presentes:
        df["indice_cluster"] = df[cols_presentes].sum(axis=1)
        log.info("Índice cluster construido desde: %s", cols_presentes)
    else:
        log.warning("No se encontraron columnas de nivel para índice cluster")
        df["indice_cluster"] = np.nan

    # cod_grupo y cod_depe2 ya vienen normalizados desde Script 05
    df["cod_grupo"] = pd.to_numeric(df["cod_grupo"], errors="coerce")
    df["cod_depe2"] = pd.to_numeric(df["cod_depe2"], errors="coerce")
    log.info("cod_grupo NaN: %d | cod_depe2 NaN: %d",
             df["cod_grupo"].isna().sum(), df["cod_depe2"].isna().sum())

    return df


# ---------------------------------------------------------------------------
# Correr un modelo OLS
# ---------------------------------------------------------------------------

def correr_ols(df_sub: pd.DataFrame, outcome: str, indice: str,
               agno: int, grado: str, asignatura: str, tipo_indice: str) -> dict | None:
    """
    Corre OLS con dummies de cod_grupo y cod_depe2.
    Retorna dict con coeficientes, SE, p-values y métricas.
    """
    import statsmodels.formula.api as smf

    cols_req = [outcome, indice, "cod_grupo", "cod_depe2"]
    df_m = df_sub[cols_req].dropna()

    if len(df_m) < 30:
        log.warning("  Pocas obs (%d) para %d %s %s — omitiendo",
                    len(df_m), agno, grado, asignatura)
        return None

    df_m["cod_grupo"] = df_m["cod_grupo"].astype(int)
    df_m["cod_depe2"] = df_m["cod_depe2"].astype(int)

    formula = (
        f"{outcome} ~ {indice} "
        f"+ C(cod_grupo, Treatment(reference={GRUPO_REF})) "
        f"+ C(cod_depe2, Treatment(reference={DEPE_REF}))"
    )

    try:
        modelo = smf.ols(formula, data=df_m).fit(
            cov_type="HC3"  # errores estándar robustos (heteroscedasticidad)
        )
    except Exception as e:
        log.error("  Error en modelo %d %s %s: %s", agno, grado, asignatura, e)
        return None

    # Extraer coeficiente del índice de crimen
    coef_crimen = modelo.params.get(indice, np.nan)
    se_crimen   = modelo.bse.get(indice, np.nan)
    pval_crimen = modelo.pvalues.get(indice, np.nan)
    tstat       = modelo.tvalues.get(indice, np.nan)

    # Significancia
    if pval_crimen < 0.001:
        sig = "***"
    elif pval_crimen < 0.01:
        sig = "**"
    elif pval_crimen < 0.05:
        sig = "*"
    elif pval_crimen < 0.1:
        sig = "+"
    else:
        sig = ""

    resultado = {
        "agno":           agno,
        "grado":          grado,
        "asignatura":     asignatura,
        "tipo_indice":    tipo_indice,
        "ventana_dias":   VENTANA_PRINCIPAL,
        "radio":          RADIO_PRINCIPAL,
        "n_obs":          int(modelo.nobs),
        "coef_crimen":    round(coef_crimen, 4),
        "se_crimen":      round(se_crimen, 4),
        "tstat_crimen":   round(tstat, 4),
        "pval_crimen":    round(pval_crimen, 4),
        "significancia":  sig,
        "r2":             round(modelo.rsquared, 4),
        "r2_adj":         round(modelo.rsquared_adj, 4),
        "rmse":           round(np.sqrt(modelo.mse_resid), 4),
        "aic":            round(modelo.aic, 2),
        "bic":            round(modelo.bic, 2),
        "log_lik":        round(modelo.llf, 3),
        "f_stat":         round(modelo.fvalue, 3) if modelo.fvalue else np.nan,
    }

    # Agregar coeficientes de controles
    for k, v in modelo.params.items():
        if k != "Intercept" and k != indice:
            resultado[f"coef_{k}"] = round(v, 4)
            resultado[f"se_{k}"]   = round(modelo.bse[k], 4)
            resultado[f"pval_{k}"] = round(modelo.pvalues[k], 4)

    return resultado


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def correr_todos_los_modelos(df: pd.DataFrame) -> tuple[list, list]:
    resultados_global  = []
    resultados_cluster = []

    indice_global  = f"indice_exp_{RADIO_PRINCIPAL}"
    indice_cluster = "indice_cluster"

    combos = sorted(df[["agno", "grado"]].drop_duplicates().values.tolist())
    log.info("Corriendo modelos para %d combos agno-grado...", len(combos))

    for agno, grado in combos:
        df_sub = df[(df["agno"] == agno) & (df["grado"] == grado)].copy()
        outcomes = OUTCOMES_POR_GRADO.get(str(grado), {})

        for asignatura, outcome_col in outcomes.items():
            if outcome_col is None:
                continue

            log.info("  %d %s %s — n=%d", agno, grado, asignatura, len(df_sub))

            # Modelo con índice global
            r = correr_ols(df_sub, outcome_col, indice_global,
                           agno, grado, asignatura, "global")
            if r:
                resultados_global.append(r)
                log.info("    Global:  β=%.4f (%s)  R²=%.3f",
                         r["coef_crimen"], r["significancia"], r["r2"])

            # Modelo con índice cluster
            r = correr_ols(df_sub, outcome_col, indice_cluster,
                           agno, grado, asignatura, "cluster")
            if r:
                resultados_cluster.append(r)
                log.info("    Cluster: β=%.4f (%s)  R²=%.3f",
                         r["coef_crimen"], r["significancia"], r["r2"])

    return resultados_global, resultados_cluster


# ---------------------------------------------------------------------------
# Tabla resumen estilo paper (agno × grado)
# ---------------------------------------------------------------------------

def construir_tabla_resumen(resultados: list) -> pd.DataFrame:
    """
    Construye tabla pivotada con coef (SE) por combinación agno-grado,
    similar al formato de las Figuras 1-14 del paper.
    """
    df = pd.DataFrame(resultados)

    # Formato: "coef*** \n(SE)"
    df["coef_fmt"] = (
        df["coef_crimen"].map("{:.4f}".format) +
        df["significancia"] +
        "\n(" + df["se_crimen"].map("{:.4f}".format) + ")"
    )

    pivot = df.pivot_table(
        index=["agno"],
        columns=["grado", "asignatura"],
        values="coef_fmt",
        aggfunc="first"
    )
    return pivot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Script 08: Regresiones OLS ===")
    log.info("Ventana: %d días | Radio: %s", VENTANA_PRINCIPAL, RADIO_PRINCIPAL)

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    df = cargar_datos()

    # 2. Correr modelos
    resultados_global, resultados_cluster = correr_todos_los_modelos(df)

    # 3. Guardar resultados detallados
    df_global  = pd.DataFrame(resultados_global)
    df_cluster = pd.DataFrame(resultados_cluster)

    path_global  = RESULTADOS_DIR / "regression_global_index.csv"
    path_cluster = RESULTADOS_DIR / "regression_cluster_index.csv"
    df_global.to_csv(path_global,   index=False)
    df_cluster.to_csv(path_cluster, index=False)
    log.info("Guardado: %s — %d modelos", path_global.name,  len(df_global))
    log.info("Guardado: %s — %d modelos", path_cluster.name, len(df_cluster))

    # 4. Tabla resumen estilo paper
    for df_res, nombre in [(df_global, "global"), (df_cluster, "cluster")]:
        tabla = construir_tabla_resumen(df_res)
        path  = RESULTADOS_DIR / f"tabla_resumen_{nombre}.csv"
        tabla.to_csv(path)
        log.info("Tabla resumen %s guardada: %s", nombre, path.name)

    # 5. Imprimir resumen ejecutivo
    log.info("\n── Resumen ejecutivo — Índice Global ──")
    resumen_cols = ["agno", "grado", "asignatura", "n_obs",
                    "coef_crimen", "se_crimen", "significancia", "r2_adj"]
    log.info("\n%s", df_global[resumen_cols].to_string(index=False))

    log.info("\n── Resumen ejecutivo — Índice Cluster ──")
    log.info("\n%s", df_cluster[resumen_cols].to_string(index=False))

    # 6. Conteo de significancia
    for nombre, df_res in [("Global", df_global), ("Cluster", df_cluster)]:
        total = len(df_res)
        sig   = (df_res["significancia"] != "").sum()
        neg   = (df_res["coef_crimen"] < 0).sum()
        log.info("%s: %d/%d modelos significativos | %d/%d coef negativos",
                 nombre, sig, total, neg, total)

    log.info("=== Script 08 completado ===")


if __name__ == "__main__":
    main()
