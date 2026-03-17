"""
01_create_crime_severity_lookup.py
Genera la tabla de referencia de ranking delictual (crime_severity_lookup.csv).

Pipeline:
    1. Lee listado_delitos.csv (datos crudos)
    2. Extrae los 98 tipos de delito únicos
    3. Aplica ranking_propuesto (1 = más grave, 98 = menos grave)
    4. Valida consistencia entre CSV y ranking
    5. Calcula score_gravedad normalizado [0, 1]
    6. Asigna nivel_gravedad (1–8) y grupo_gravedad
    7. Guarda config/crime_severity_lookup.csv

Ejecución:
    python scripts/01_create_crime_severity_lookup.py

Notas:
    - El ranking puede editarse manualmente en este archivo.
    - Después de editar, correr 02_validate_and_score_lookup.py para
      revalidar y recomputar scores sin tocar los datos crudos. --> NO MODIFICAR DIRECTAMENTE EL SCRIPT 02_VALIDATE_AND_SCORE_LOOKUP.PY
"""

import logging
import sys

import pandas as pd

from project_paths import (
    CRIME_SEVERITY_LOOKUP_PATH,
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
# RANKING OFICIAL 
# ---------------------------------------------------------------------------
# Criterios de ordenamiento:
#   Nivel 1 — Violencia letal extrema           (ranks  1 –  5)
#   Nivel 2 — Violencia grave                   (ranks  6 – 12)
#   Nivel 3 — Armas y robos                     (ranks 13 – 26)
#   Nivel 4 — Propiedad, lesiones y sexuales    (ranks 27 – 46)
#   Nivel 5 — Drogas                            (ranks 47 – 52)
#   Nivel 6 — Económicos e institucionales      (ranks 53 – 65)
#   Nivel 7 — Orden público e infracciones      (ranks 66 – 81)
#   Nivel 8 — Incivilidades                     (ranks 82 – 98)
# ---------------------------------------------------------------------------

RANKING_PROPUESTO = {
    # ── Nivel 1 · Violencia letal extrema ─────────────────────────────────
    "Homicidio":                                                          1,
    "Secuestro":                                                          2,
    "Violaciones":                                                        3,
    "Trata de personas":                                                  4,
    "Tráfico de armas":                                                   5,

    # ── Nivel 2 · Violencia grave ──────────────────────────────────────────
    "Robo con violencia o intimidación":                                  6,
    "Lesiones menos graves, graves o gravísimas":                         7,
    "Violencia intrafamiliar a mujer":                                    8,
    "Violencia intrafamiliar a NNA":                                      9,
    "Violencia intrafamiliar a adulto mayor (>=65a)":                    10,
    "Violencia intrafamiliar a hombre":                                  11,
    "Utilización de artefactos artesanales, bombas molotov o similares": 12,

    # ── Nivel 3 · Armas y robos ────────────────────────────────────────────
    "Posesión, tenencia o porte ilegal de armas":                        13,
    "Posesión, tenencia o porte ilegal de munición":                     14,
    "Porte de arma blanca":                                              15,
    "Portar elementos destinados a cometer delito de robo":              16,
    "Portar objetos robados o hurtados":                                 17,
    "Receptación":                                                       18,
    "Robo de vehículo motorizado por sorpresa, violencia o intimidación":19,
    "Robo vehículos motorizados":                                        20,
    "Robo en lugar habitado":                                            21,
    "Robo en lugar no habitado":                                         22,
    "Robo de objeto de o desde vehículo":                                23,
    "Otros robos con fuerza":                                            24,
    "Robo por sorpresa":                                                 25,
    "Robo frustrado":                                                    26,

    # ── Nivel 4 · Propiedad, lesiones y delitos sexuales ──────────────────
    "Hurto":                                                             27,
    "Abigeato":                                                          28,
    "Incendios y otros estragos":                                        29,
    "Violación de morada":                                               30,
    "Lesiones leves":                                                    31,
    "Riña pública":                                                      32,
    "Amenazas contra la autoridad":                                      33,
    "Apremios ilegítimos":                                               34,
    "Maltrato obra":                                                     35,
    "Auxilio suicidio":                                                  36,
    "Vulneración de menores":                                            37,
    "Vulneración de derechos":                                           38,
    "Menor abandono o maltrato":                                         39,
    "Abandono persona u hogar":                                          40,
    "Abuso sexual de menor de 14 años":                                  41,
    "Abuso sexual de mayor de 14 años y menor a 18 años":               42,
    "Otros delitos sexuales":                                            43,
    "Delitos informáticos":                                              44,
    "Delitos de carácter económico":                                     45,
    "Delitos contra la propiedad intelectual":                           46,

    # ── Nivel 5 · Drogas ───────────────────────────────────────────────────
    "Tráfico de droga":                                                  47,
    "Microtráfico de droga":                                             48,
    "Cultivo de droga":                                                  49,
    "Consumo de droga":                                                  50,
    "Porte de droga":                                                    51,
    "Otras infracciones ley 20.000":                                     52,

    # ── Nivel 6 · Económicos e institucionales ─────────────────────────────
    "Falsificación, perjurio y falso testimonio":                        53,
    "Soborno, cohecho y tráfico de influencias":                         54,
    "Delitos funcionarios públicos":                                     55,
    "Exacciones legales":                                                56,
    "Tráfico de emigrantes":                                             57,
    "Crímenes y delitos derechos constitucionales":                      58,
    "Aborto":                                                            59,
    "Cuasidelito":                                                       60,
    "Daños":                                                             61,
    "Daño falta":                                                        62,
    "Objeto contundente a vehículo":                                     63,
    "Hallazgo de armamento":                                             64,
    "Hallazgo de droga":                                                 65,

    # ── Nivel 7 · Orden público e infracciones ────────────────────────────
    "Hallazgo de vehículo":                                              66,
    "Hallazgo de cuerpo y otras muertes":                                67,
    "Presunta desgracia":                                                68,
    "Orden de aprehensión":                                              69,
    "Quebrantamiento de sentencia, fuga y evasión":                      70,
    "Hechos y faltas contra policías y fiscales":                        71,
    "Desacato o resistencia a la autoridad":                             72,
    "Desórdenes":                                                        73,
    "Violencia en los estadios":                                         74,
    "Salud pública":                                                     75,
    "Protección animal y medio ambiente":                                76,
    "Animales (sueltos o responsabilidad del dueño)":                    77,
    "Juegos y loterías ilegales y ley de casinos":                       78,
    "Infracciones ley electoral":                                        79,
    "Ley de tránsito":                                                   80,
    "Ley de alcoholes":                                                  81,

    # ── Nivel 8 · Incivilidades ────────────────────────────────────────────
    "Ley de tabaco":                                                     82,
    "Otras infracciones juzgado de policía local":                       83,
    "Otros delitos de la ley de control de armas":                       84,
    "Otros hechos y faltas":                                             85,
    "Menores (delitos, faltas y otros hechos similares)":                86,
    "Mala conducta menor":                                               87,
    "Amenaza (incivilidades)":                                           88,
    "Disensiones domésticas":                                            89,
    "Ebriedad":                                                          90,
    "Conducción ebriedad o influencia del alcohol":                      91,
    "Consumo de alcohol en vía pública":                                 92,
    "Ruidos molestos":                                                   93,
    "Comercio ambulante o clandestino":                                  94,
    "Especies abandonadas en vía pública":                               95,
    "Pensiones alimenticias":                                            96,
    "Ofensas al pudor o ultraje público buenas costumbres":              97,
    "Extranjería":                                                       98,
}


# ---------------------------------------------------------------------------
# Asignación de niveles y grupos
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

# Construir lookup rápido rank → (nivel, grupo)
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
# I/O helpers
# ---------------------------------------------------------------------------

def leer_csv_con_encoding(path) -> pd.DataFrame:
    """Intenta UTF-8 primero; si falla, cae a latin-1."""
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            log.info("CSV leído con encoding '%s' — %d filas", enc, len(df))
            return df
        except UnicodeDecodeError:
            log.warning("Encoding '%s' falló, intentando siguiente...", enc)
    raise ValueError(f"No se pudo leer {path} con utf-8 ni latin-1.")


# ---------------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------------

def validar_ranking_interno():
    """Verifica que el ranking sea 1..98 sin duplicados ni saltos."""
    ranks = sorted(RANKING_PROPUESTO.values())
    esperado = list(range(1, len(RANKING_PROPUESTO) + 1))
    if ranks != esperado:
        duplicados = [r for r in ranks if ranks.count(r) > 1]
        faltantes  = sorted(set(esperado) - set(ranks))
        msg = "RANKING_PROPUESTO inválido."
        if duplicados:
            msg += f" Duplicados: {set(duplicados)}."
        if faltantes:
            msg += f" Ranks faltantes: {faltantes}."
        raise ValueError(msg)
    log.info("Ranking interno OK — %d delitos, ranks 1–%d", len(ranks), max(ranks))


def validar_contra_csv(delitos_csv: list[str]):
    """Compara los delitos del CSV con los del ranking y lanza error si hay diferencias."""
    set_csv     = set(delitos_csv)
    set_ranking = set(RANKING_PROPUESTO.keys())

    faltan = sorted(set_csv - set_ranking)
    sobran = sorted(set_ranking - set_csv)

    if faltan:
        log.error("Delitos en CSV pero AUSENTES en ranking (%d):", len(faltan))
        for d in faltan:
            log.error("  FALTA: '%s'", d)

    if sobran:
        log.warning("Delitos en ranking pero AUSENTES en CSV (%d):", len(sobran))
        for d in sobran:
            log.warning("  SOBRA: '%s'", d)

    if faltan:
        raise ValueError(
            f"{len(faltan)} delito(s) del CSV no tienen ranking asignado. "
            "Agrégalos a RANKING_PROPUESTO antes de continuar."
        )

    if sobran:
        log.warning(
            "Hay %d delito(s) en el ranking que no aparecen en el CSV. "
            "Serán incluidos en el lookup igualmente.", len(sobran)
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== Script 01: Crear crime_severity_lookup ===")

    # 1. Validar ranking antes de tocar el CSV
    validar_ranking_interno()

    # 2. Leer CSV
    if not RAW_CRIME_DATA_PATH.exists():
        log.error("No se encontró el archivo: %s", RAW_CRIME_DATA_PATH)
        sys.exit(1)

    df = leer_csv_con_encoding(RAW_CRIME_DATA_PATH)

    # 3. Validar columnas requeridas
    required_cols = {"fecha", "GEOCODIGO", "delito", "frecuencia"}
    missing_cols  = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(missing_cols)}")

    # 4. Extraer delitos únicos del CSV
    delitos_csv = (
        df["delito"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )
    log.info("Delitos únicos en CSV: %d", len(delitos_csv))
    log.info("Cantidad de delitos en CSV:     %d", len(delitos_csv))
    log.info("Cantidad de delitos en ranking: %d", len(RANKING_PROPUESTO))

    # 5. Validar consistencia CSV ↔ ranking
    validar_contra_csv(delitos_csv)

    # 6. Construir tabla de severidad
    severity = pd.DataFrame({"delito": list(RANKING_PROPUESTO.keys())})
    severity["rank_gravedad"] = severity["delito"].map(RANKING_PROPUESTO).astype(int)

    max_rank = severity["rank_gravedad"].max()   # siempre 98
    severity["score_gravedad"] = (max_rank - severity["rank_gravedad"]) / (max_rank - 1)

    niveles = severity["rank_gravedad"].apply(asignar_nivel_y_grupo)
    severity["nivel_gravedad"] = niveles.apply(lambda x: x[0])
    severity["grupo_gravedad"] = niveles.apply(lambda x: x[1])

    severity["comentario"] = pd.NA

    severity = severity.sort_values("rank_gravedad").reset_index(drop=True)

    # 7. Guardar
    CRIME_SEVERITY_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    severity.to_csv(CRIME_SEVERITY_LOOKUP_PATH, index=False, encoding="utf-8")

    log.info("Lookup guardado en: %s", CRIME_SEVERITY_LOOKUP_PATH)
    log.info("Total delitos: %d", len(severity))

    # 8. Resumen por nivel
    resumen = (
        severity
        .groupby(["nivel_gravedad", "grupo_gravedad"])
        .agg(n_delitos=("delito", "count"),
             rank_min=("rank_gravedad", "min"),
             rank_max=("rank_gravedad", "max"),
             score_min=("score_gravedad", "min"),
             score_max=("score_gravedad", "max"))
        .reset_index()
    )
    log.info("\n%s", resumen.to_string(index=False))


if __name__ == "__main__":
    main()
