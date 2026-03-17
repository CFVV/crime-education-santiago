"""
02_validate_and_score_lookup.py
Valida y recomputa el archivo crime_severity_lookup.csv.

Objetivo:
    Permite que se edite manualmente el ranking en
    crime_severity_lookup.csv (ej: mover un delito de rank)
    y luego corra este script para:

        1. Verificar integridad del ranking editado
        2. Recomputar score_gravedad normalizado
        3. Recomputar nivel_gravedad y grupo_gravedad
        4. Guardar el lookup actualizado (sobreescribe el archivo)
        5. Generar un reporte de validación legible en consola

Casos de uso:
    - Después de editar rank_gravedad manualmente en el CSV
    - Después de agregar o eliminar un delito del lookup
    - Como verificación de rutina antes de correr el Script 03

Ejecución:
    python scripts/02_validate_and_score_lookup.py

    # Solo validar sin sobreescribir:
    python scripts/02_validate_and_score_lookup.py --dry-run
"""

import argparse
import logging
import sys

import pandas as pd

from project_paths import CRIME_SEVERITY_LOOKUP_PATH

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
# Configuración de niveles
# (debe mantenerse sincronizada con Script 01)
# ---------------------------------------------------------------------------
NIVEL_CONFIG = {
    1: (range(1,   6),  "violencia_letal_extrema"),
    2: (range(6,  13),  "violencia_grave"),
    3: (range(13, 27),  "armas_y_robos"),
    4: (range(27, 47),  "propiedad_lesiones_sexuales"),
    5: (range(47, 53),  "drogas"),
    6: (range(53, 66),  "economicos_institucionales"),
    7: (range(66, 82),  "orden_publico_infracciones"),
    8: (range(82, 99),  "incivilidades"),
}

_RANK_TO_NIVEL = {}
for nivel, (rng, grupo) in NIVEL_CONFIG.items():
    for r in rng:
        _RANK_TO_NIVEL[r] = (nivel, grupo)


def asignar_nivel_y_grupo(rank: int) -> tuple[int, str]:
    try:
        return _RANK_TO_NIVEL[rank]
    except KeyError:
        raise ValueError(f"Rank fuera de rango [1-98]: {rank}")


# ---------------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"delito", "rank_gravedad", "score_gravedad", "nivel_gravedad",
                 "grupo_gravedad", "comentario"}


def validar_columnas(df: pd.DataFrame) -> list[str]:
    errores = []
    faltantes = REQUIRED_COLS - set(df.columns)
    if faltantes:
        errores.append(f"Columnas faltantes en el lookup: {sorted(faltantes)}")
    return errores


def validar_ranking(df: pd.DataFrame) -> list[str]:
    """Verifica que rank_gravedad sea 1..N sin duplicados ni saltos."""
    errores = []
    ranks = sorted(df["rank_gravedad"].dropna().astype(int).tolist())
    n = len(ranks)
    esperado = list(range(1, n + 1))

    if ranks != esperado:
        duplicados = sorted({r for r in ranks if ranks.count(r) > 1})
        faltantes  = sorted(set(esperado) - set(ranks))
        extras     = sorted(set(ranks) - set(esperado))

        if duplicados:
            errores.append(f"Ranks duplicados: {duplicados}")
        if faltantes:
            errores.append(f"Ranks faltantes en la secuencia: {faltantes}")
        if extras:
            errores.append(f"Ranks fuera del rango esperado 1–{n}: {extras}")

    return errores


def validar_delitos_unicos(df: pd.DataFrame) -> list[str]:
    errores = []
    duplicados = df[df.duplicated(subset="delito", keep=False)]["delito"].unique()
    if len(duplicados) > 0:
        errores.append(f"Delitos duplicados en lookup: {sorted(duplicados)}")
    return errores


def validar_nulos(df: pd.DataFrame) -> list[str]:
    errores = []
    for col in ["delito", "rank_gravedad"]:
        n_null = df[col].isna().sum()
        if n_null > 0:
            errores.append(f"Columna '{col}' tiene {n_null} valor(es) nulo(s)")
    return errores


# ---------------------------------------------------------------------------
# Recomputar scores y niveles
# ---------------------------------------------------------------------------

def recomputar(df: pd.DataFrame) -> pd.DataFrame:
    """Recomputa score_gravedad, nivel_gravedad y grupo_gravedad desde rank_gravedad."""
    df = df.copy()
    df["rank_gravedad"] = df["rank_gravedad"].astype(int)

    max_rank = df["rank_gravedad"].max()
    min_rank = df["rank_gravedad"].min()

    if max_rank == min_rank:
        raise ValueError("Todos los ranks son iguales — no es posible normalizar.")

    df["score_gravedad"] = (max_rank - df["rank_gravedad"]) / (max_rank - 1)

    niveles = df["rank_gravedad"].apply(asignar_nivel_y_grupo)
    df["nivel_gravedad"] = niveles.apply(lambda x: x[0])
    df["grupo_gravedad"] = niveles.apply(lambda x: x[1])

    df = df.sort_values("rank_gravedad").reset_index(drop=True)

    # Preservar columnas extra que el usuario haya agregado
    cols_base = ["delito", "rank_gravedad", "score_gravedad",
                 "nivel_gravedad", "grupo_gravedad", "comentario"]
    cols_extra = [c for c in df.columns if c not in cols_base]
    return df[cols_base + cols_extra]


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def imprimir_reporte(df: pd.DataFrame):
    log.info("\n" + "=" * 60)
    log.info("REPORTE DE VALIDACIÓN — crime_severity_lookup")
    log.info("=" * 60)
    log.info("Total delitos: %d", len(df))
    log.info("Rank mínimo:   %d | Score: %.4f", df['rank_gravedad'].min(), df['score_gravedad'].max())
    log.info("Rank máximo:   %d | Score: %.4f", df['rank_gravedad'].max(), df['score_gravedad'].min())

    log.info("\nDistribución por nivel:")
    resumen = (
        df.groupby(["nivel_gravedad", "grupo_gravedad"])
        .agg(
            n_delitos=("delito", "count"),
            rank_min=("rank_gravedad", "min"),
            rank_max=("rank_gravedad", "max"),
            score_max=("score_gravedad", "max"),
            score_min=("score_gravedad", "min"),
        )
        .reset_index()
    )
    for _, row in resumen.iterrows():
        log.info(
            "  Nivel %d (%s): %2d delitos | ranks %2d–%2d | scores %.4f–%.4f",
            row["nivel_gravedad"],
            row["grupo_gravedad"],
            row["n_delitos"],
            row["rank_min"],
            row["rank_max"],
            row["score_min"],
            row["score_max"],
        )

    log.info("\nTop 10 delitos más graves:")
    top = df.nsmallest(10, "rank_gravedad")[["rank_gravedad", "delito", "score_gravedad", "grupo_gravedad"]]
    for _, row in top.iterrows():
        log.info("  [%2d] %.4f | %-55s | %s",
                 row["rank_gravedad"], row["score_gravedad"],
                 row["delito"], row["grupo_gravedad"])

    log.info("\nTop 10 delitos menos graves:")
    bottom = df.nlargest(10, "rank_gravedad")[["rank_gravedad", "delito", "score_gravedad", "grupo_gravedad"]]
    for _, row in bottom.iterrows():
        log.info("  [%2d] %.4f | %-55s | %s",
                 row["rank_gravedad"], row["score_gravedad"],
                 row["delito"], row["grupo_gravedad"])

    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Valida y recomputa crime_severity_lookup.csv")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo validar y mostrar reporte, sin sobreescribir el archivo."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log.info("=== Script 02: Validar y recomputar lookup ===")
    if args.dry_run:
        log.info("Modo: DRY-RUN (no se sobreescribirá el archivo)")

    # 1. Verificar que el archivo existe
    if not CRIME_SEVERITY_LOOKUP_PATH.exists():
        log.error("No se encontró el lookup en: %s", CRIME_SEVERITY_LOOKUP_PATH)
        log.error("Corre primero: python scripts/01_create_crime_severity_lookup.py")
        sys.exit(1)

    # 2. Leer lookup
    df = pd.read_csv(CRIME_SEVERITY_LOOKUP_PATH, encoding="utf-8")
    log.info("Lookup leído: %d filas, %d columnas", len(df), len(df.columns))

    # 3. Validaciones
    errores = []
    errores += validar_columnas(df)
    errores += validar_nulos(df)
    errores += validar_delitos_unicos(df)
    errores += validar_ranking(df)

    if errores:
        log.error("\n%d error(es) encontrado(s):", len(errores))
        for e in errores:
            log.error("  ✗ %s", e)
        log.error("Corrige el lookup manualmente y vuelve a correr este script.")
        sys.exit(1)

    log.info("Validaciones pasadas ✓")

    # 4. Recomputar
    df_recomputed = recomputar(df)

    # 5. Reporte
    imprimir_reporte(df_recomputed)

    # 6. Guardar (si no es dry-run)
    if not args.dry_run:
        df_recomputed.to_csv(CRIME_SEVERITY_LOOKUP_PATH, index=False, encoding="utf-8")
        log.info("Lookup actualizado guardado en: %s", CRIME_SEVERITY_LOOKUP_PATH)
    else:
        log.info("Dry-run: archivo NO modificado.")


if __name__ == "__main__":
    main()
