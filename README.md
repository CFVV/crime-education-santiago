# Crime and Educational Performance in Santiago (2013–2019)

Replication and extension of the analysis pipeline for:  
*"Impact of Crime Exposure on Educational Performance in Santiago de Chile (2013–2019)"*

## Requirements
```bash
pip install pandas geopandas shapely pyproj statsmodels pyarrow
```

## Data not included

The following files are required but not included in this repository:

| File | Source |
|------|--------|
| `data/raw/listado_delitos.csv` | CEAD / Carabineros de Chile |
| `data/raw/shapefiles/rm_manzanas_geocodigo.shp` | INE — Censo 2017 |
| `data/processed/simce_completo_5.dta` | Agencia de Calidad de la Educacion |

Contact the research team for access.

The MINEDUC school directories (2013–2019) are included in `data/raw/directorios_mineduc/`.

## Pipeline
```bash
python scripts/01_create_crime_severity_lookup.py
python scripts/02_validate_and_score_lookup.py
python scripts/03_create_crime_index_by_geocode.py
python scripts/04_dataprep_establecimientos.py
python scripts/05_dataprep_education.py
python scripts/06_spatial_buffers.py   # requiere listado_delitos.csv y shapefiles
python scripts/07_final_table.py
python scripts/08_regression_analysis.py
```

See `docs/documentacion_pipeline.docx` for full technical documentation.  
See `notebooks/explore_tables.ipynb` for a tour of the output tables.
