"""
project_paths.py
=================
Centralizador de rutas del proyecto. Todos los scripts importan sus paths
desde aquí. Si cambia la estructura de carpetas, este es el único archivo
que necesita modificarse.

Ubicación esperada: crime_education_project/scripts/project_paths.py

Pipeline completo (orden de ejecución):
    01  create_crime_severity_lookup      → config/crime_severity_lookup.csv
    02  validate_and_score_lookup         → config/crime_severity_lookup.csv  (actualizado)
    03  create_crime_index_by_geocode     → intermediate/crime_index_geocode_daily.csv
    04  dataprep_establecimientos         → processed/schools_clean_2014_2019.parquet
    05  dataprep_schools                  → processed/schools_clean_2013_2019_RM.parquet
                                            processed/school_locations_RM.parquet
    06  dataprep_simce_base               → processed/data_education_2014_2019.parquet
    07  dataprep_education                → processed/data_education_2013_2019_RM.parquet
    08  spatial_buffers                   → intermediate/crime_exposure_400m.parquet
                                            intermediate/crime_exposure_1km.parquet
    09  final_table                       → processed/tabla_final_escuelas_crimen.parquet
    10  regression_analysis               → resultados/regression_global_index.csv
                                            resultados/regression_cluster_index.csv
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Raíz del proyecto
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Directorios base
# ---------------------------------------------------------------------------
CONFIG_DIR            = PROJECT_ROOT / "config"
DATA_DIR              = PROJECT_ROOT / "data"
RAW_DATA_DIR          = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
PROCESSED_DATA_DIR    = DATA_DIR / "processed"
RESULTS_DIR           = DATA_DIR / "resultados"
DOCS_DIR              = PROJECT_ROOT / "docs"
OUTPUTS_DIR           = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Script 01 / 02 — Lookup de severidad (config, editable manualmente)
# ---------------------------------------------------------------------------
RAW_CRIME_DATA_PATH        = RAW_DATA_DIR / "listado_delitos.csv"
CRIME_SEVERITY_LOOKUP_PATH = CONFIG_DIR   / "crime_severity_lookup.csv"

# ---------------------------------------------------------------------------
# Script 03 — Índice de criminalidad por geocódigo
# ---------------------------------------------------------------------------
CRIME_INDEX_DAILY_PATH    = INTERMEDIATE_DATA_DIR / "crime_index_geocode_daily.csv"
CRIME_INDEX_MONTHLY_PATH  = INTERMEDIATE_DATA_DIR / "crime_index_geocode_monthly.csv"
CRIME_DETAIL_DAILY_PATH   = INTERMEDIATE_DATA_DIR / "crime_detail_geocode_daily.csv"
CRIME_DETAIL_MONTHLY_PATH = INTERMEDIATE_DATA_DIR / "crime_detail_geocode_monthly.csv"

# ---------------------------------------------------------------------------
# Script 04 — Preparación completa de establecimientos 2013–2019 RM
# (inputs: Directorios_mineduc/Directorio_oficial_EE_YYYY.csv, 2013–2019)
# ---------------------------------------------------------------------------
DIRECTORIOS_MINEDUC_DIR = RAW_DATA_DIR / "Directorios_mineduc"
SCHOOLS_OUTPUT_PATH     = PROCESSED_DATA_DIR / "schools_clean_2013_2019_RM.parquet"
SCHOOL_LOCATIONS_PATH   = PROCESSED_DATA_DIR / "school_locations_RM.parquet"

# ---------------------------------------------------------------------------
# Script 05 — Preparación de datos educación 2013–2019 RM (SIMCE)
# (inputs: simce_completo_5.dta + Script 04)
# ---------------------------------------------------------------------------
SIMCE_RAW_PATH        = PROCESSED_DATA_DIR / "simce_completo_5.dta"
EDUCATION_OUTPUT_PATH = PROCESSED_DATA_DIR / "data_education_2013_2019_RM.parquet"

# ---------------------------------------------------------------------------
# Script 06 — Buffers espaciales (400m y 1km)
# ---------------------------------------------------------------------------
CENSUS_BLOCKS_PATH       = RAW_DATA_DIR / "shapefiles" / "rm_manzanas_geocodigo.shp"
CRIME_BUFFER_400M_PATH   = INTERMEDIATE_DATA_DIR / "crime_exposure_400m.parquet"
CRIME_BUFFER_1KM_PATH    = INTERMEDIATE_DATA_DIR / "crime_exposure_1km.parquet"
CRIME_GEOREF_PATH        = INTERMEDIATE_DATA_DIR / "crime_index_georef.parquet"
CRIME_DETAIL_GEOREF_PATH = INTERMEDIATE_DATA_DIR / "crime_detail_georef.parquet"

# ---------------------------------------------------------------------------
# Script 07 — Tabla objetivo final
# ---------------------------------------------------------------------------
FINAL_TABLE_PATH = PROCESSED_DATA_DIR / "tabla_final_escuelas_crimen.parquet"

# ---------------------------------------------------------------------------
# Script 08 — Regresión
# ---------------------------------------------------------------------------
REGRESSION_GLOBAL_PATH  = RESULTS_DIR / "regression_global_index.csv"
REGRESSION_CLUSTER_PATH = RESULTS_DIR / "regression_cluster_index.csv"
SUMMARY_GLOBAL_PATH     = RESULTS_DIR / "tabla_resumen_global.csv"
SUMMARY_CLUSTER_PATH    = RESULTS_DIR / "tabla_resumen_cluster.csv"
