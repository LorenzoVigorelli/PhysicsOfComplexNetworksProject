== SCI (TSV) – prime righe ==
| user_loc | fr_loc | scaled_sci |
|----------|--------|------------|
| ABW      | ABW    | 11264841.0 |
| ABW      | AGO1   | 38.0       |
| ABW      | AGO10  | 34.0       |
| ABW      | AGO11  | 32.0       |
| ABW      | AGO12  | 23.0       |

Schema:
user_loc      string[python]
fr_loc        string[python]
scaled_sci           float64
dtype: object

Numero righe: 63,824,121
== Mapping livelli – prime righe ==
| location_code | level_type |
|---------------|------------|
| AND           | country    |
| ATG           | country    |
| ABW           | country    |
| BHS           | country    |
| BRB           | country    |


Schema:
location_code    string[python]
level_type       string[python]
dtype: object

Numero righe: 8,008

== level_type value_counts ==
level_type
county     3229
gadm1      1839
nuts3      1522
gadm2      1370
country      48
Name: count, dtype: Int64

== Copertura NODI ==
- total_unique_codes: 7989
- mapped_unique_codes: 7984
- unmapped_unique_codes: 5
- node_coverage_pct: 99.93741394417324

Esempi codici NON mappati (5):
| location_code |
|---------------|
| ASM           |
| GUM           |
| MNP           |
| MUS1          |
| VIR           |


== Copertura ARCHI ==
- total_rows: 63824121
- valid_rows_both_mapped: 63744256
- edge_coverage_pct: 99.87486705849031
[AVVISO] Colonna 'country_ISO3' assente nel mapping: copertura per paese non calcolata.

== Copertura PER PAESE ==
Mapping senza 'country_ISO3' → se vuoi questa sezione, aggiungi la colonna a df_map.
Layer trovati in data/gadm_410.gpkg:
 1. gadm_410

Colonne disponibili in NUTS3 GeoJSON:
['LEVL_CODE', 'NUTS_ID', 'CNTR_CODE', 'NAME_LATN', 'NUTS_NAME', 'MOUNT_TYPE', 'URBN_TYPE', 'COAST_TYPE', 'geometry']

| LEVL_CODE | NUTS_ID | CNTR_CODE | NAME_LATN           | NUTS_NAME           | MOUNT_TYPE | URBN_TYPE | COAST_TYPE | geometry                                         |
|-----------|---------|-----------|---------------------|---------------------|------------|-----------|------------|-------------------------------------------------|
| 3         | CZ052   | CZ        | Královéhradecký kraj | Královéhradecký kraj | 4          | 2         | 3          | POLYGON ((16.10732 50.66207, 16.33255 50.59246... |
| 3         | CZ053   | CZ        | Pardubický kraj     | Pardubický kraj     | 4          | 3         | 3          | POLYGON ((16.8042 49.59881, 16.39363 49.58061,... |
| 3         | CZ063   | CZ        | Kraj Vysočina       | Kraj Vysočina       | 4          | 3         | 3          | POLYGON ((16.39363 49.58061, 16.25967 49.27462... |
| 3         | CZ064   | CZ        | Jihomoravský kraj   | Jihomoravský kraj   | 4          | 2         | 3          | POLYGON ((17.15943 49.27462, 17.27319 49.05789... |
| 3         | CZ071   | CZ        | Olomoucký kraj      | Olomoucký kraj      | 2          | 2         | 3          | POLYGON ((17.4296 50.25451, 17.17647 49.95354,... |

| intptlat   | countyfp_nozero | countyns | stusab | csafp | state_name   | aland       | geoid | namelsad         | countyfp | classfp | lsad | name     | funcstat | metdivfp | cbsafp | intptlon    | statefp | mtfcc | geometry                                         |
|------------|-----------------|----------|--------|-------|--------------|-------------|-------|-----------------|----------|---------|------|----------|----------|----------|--------|-------------|---------|-------|-------------------------------------------------|
| +43.0066030| 135             | 01266975 | SD     | None  | South Dakota | 1349873585  | 46135 | Yankton County  | 135      | H1      | 06   | Yankton  | A        | None     | 49460  | -097.3883614 | 46      | G4020 | POLYGON ((-97.51843 43.16903, -97.49807 43.169... |
| +41.5929185| 49              | 00277289 | CA     | None  | California   | 10225096402 | 06049 | Modoc County    | 049      | H1      | 06   | Modoc    | A        | None     | None   | -120.7183704 | 06      | G4020 | POLYGON ((-121.4489 41.47281, -121.44891 41.47... |
| +32.2388026| 235             | 00347593 | GA     | None  | Georgia      | 645583957   | 13235 | Pulaski County  | 235      | H1      | 06   | Pulaski  | A        | None     | None   | -083.4818454 | 13      | G4020 | POLYGON ((-83.6065 32.26751, -83.60621 32.2756... |
| +39.1642619| 13              | 00424208 | IL     | 476   | Illinois     | 657422422   | 17013 | Calhoun County  | 013      | H1      | 06   | Calhoun  | A        | None     | 41180  | -090.6662949 | 17      | G4020 | POLYGON ((-90.71598 39.19147, -90.716 39.19155... |
| +30.2064437| 5               | 00558403 | LA     | None  | Louisiana    | 751259388   | 22005 | Ascension Parish| 005      | H1      | 15   | Ascension| A        | None     | 12940  | -090.9125023 | 22      | G4020 | POLYGON ((-91.0122 30.33565, -91.0118 30.33575... |

[INFO] Caricato layer='gadm_410' con 356,508 feature
[INFO] CRS: EPSG:4326
[INFO] Colonne disponibili:
['UID', 'GID_0', 'NAME_0', 'VARNAME_0', 'GID_1', 'NAME_1', 'VARNAME_1', 'NL_NAME_1', 'ISO_1', 'HASC_1', 'CC_1', 'TYPE_1', 'ENGTYPE_1', 'VALIDFR_1', 'GID_2', 'NAME_2', 'VARNAME_2', 'NL_NAME_2', 'HASC_2', 'CC_2', 'TYPE_2', 'ENGTYPE_2', 'VALIDFR_2', 'GID_3', 'NAME_3', 'VARNAME_3', 'NL_NAME_3', 'HASC_3', 'CC_3', 'TYPE_3', 'ENGTYPE_3', 'VALIDFR_3', 'GID_4', 'NAME_4', 'VARNAME_4', 'CC_4', 'TYPE_4', 'ENGTYPE_4', 'VALIDFR_4', 'GID_5', 'NAME_5', 'CC_5', 'TYPE_5', 'ENGTYPE_5', 'GOVERNEDBY', 'SOVEREIGN', 'DISPUTEDBY', 'REGION', 'VARREGION', 'COUNTRY', 'CONTINENT', 'SUBCONT', 'geometry']
[INFO] Colonna codice ADM2 individuata: 'GID_2'
[INFO] Righe ADM2 (non-NaN su GID_2): 356,508
[INFO] Codici ADM2 unici: 47,218
[INFO] Geometrie (conteggio per tipo): {'MultiPolygon': 356508}

== Anteprima ADM2 (prime 5 righe) ==
| GID_2     | NAME_0      | NAME_1     | NAME_2   | GID_0 | GID_1   |
|-----------|-------------|------------|----------|-------|---------|
| AFG.1.1_1 | Afghanistan | Badakhshan | Baharak  | AFG   | AFG.1_1 |
| AFG.1.2_1 | Afghanistan | Badakhshan | Darwaz   | AFG   | AFG.1_1 |
| AFG.1.3_1 | Afghanistan | Badakhshan | Fayzabad | AFG   | AFG.1_1 |
| AFG.1.4_1 | Afghanistan | Badakhshan | Ishkashim| AFG   | AFG.1_1 |
| AFG.1.5_1 | Afghanistan | Badakhshan | Jurm     | AFG   | AFG.1_1 |


== Missing count su colonne comuni ==
|        | missing |
|--------|---------|
| NAME_0 | 0       |
| NAME_1 | 0       |
| NAME_2 | 0       |
| GID_0  | 0       |
| GID_1  | 0       |
| GID_2  | 0       |


[INFO] Target codes selezionati: 7,936 (tipi: ['COUNTY', 'GADM1', 'GADM2', 'NUTS3'])
Skipping field geo_point_2d: unsupported OGR type: 3
[GADM2] Caricate 47,218 unità ADM2 uniche da data/gadm_410.gpkg
NUTS pts: 1522 — US counties pts: 3233 — GADM1 pts: 3662 — GADM2 pts: 47218
NUTS3 selezionati: 1522
GADM2 selezionati: 82
GADM1 selezionati (prima del filtro): 1738
COUNTY selezionati: 3225
| nodeID | nodeLabel | nodeName     | latitude   | longitude  | DatasetDiOrigine |
|--------|-----------|--------------|------------|------------|------------------|
| 1      | AGO.10_1  | Huíla        | -13.869244 | 15.319441  | GADM1            |
| 2      | AGO.11_1  | Luanda       | -8.789466  | 13.385529  | GADM1            |
| 3      | AGO.12_1  | Lunda Norte  | -8.303799  | 21.399342  | GADM1            |
| 4      | AGO.13_1  | Lunda Sul    | -10.888833 | 19.163756  | GADM1            |
| 5      | AGO.14_1  | Malanje      | -9.323313  | 15.771779  | GADM1            |


Totale nodi: 6,567
Salvato: node_list.csv (colonne: ['nodeID', 'nodeLabel', 'nodeName', 'latitude', 'longitude', 'DatasetDiOrigine'])


== Copertura FB → NODI per TIPO ==
| dataset_type | fb_unique_nodes | matched_nodes | coverage_pct | unmatched_nodes |
|--------------|-----------------|---------------|--------------|-----------------|
| COUNTY       | 3225            | 3225          | 100.000000   | 0               |
| GADM1        | 1819            | 1738          | 95.547004    | 81              |
| GADM2        | 1370            | 82            | 5.985401     | 1288            |
| NUTS3        | 1522            | 1522          | 100.000000   | 0               |
