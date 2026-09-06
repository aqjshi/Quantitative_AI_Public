Chapters 2-4 are currently available as public demonstration of our work.



Implicit Environemnt Setup
1. Setup python and IDE on Machine
2. Setup PgAdmin4, with the SysAdmin Permissions. 
3. Acquire a .env from an admin. Keep a copy of your .env, swap it in to replace admins'. 

for admin:
Virtual Environment Setup (expect 5 minutes and 4 GB)
pip freeze | ForEach-Object { ($_ -split '==')[0] } > requirements.txt


if windows:
python -m venv venv

./venv/scripts/activate.ps1
pip install -r requirements.txt

else:
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt




Cleaning The Database:

SELECT table_schema, table_name 
FROM information_schema.tables 
WHERE table_type = 'BASE TABLE'
ORDER BY table_schema, table_name;

**PRUNE PUBLIC TABLES ONLY**

Checking size of table 
SELECT pg_size_pretty(pg_total_relation_size('{table_name}'));


If a table is messed up, you have to regularly prune out corrupted table. Contact admin for assistance. 





Precomputation and Universe Seeding (run once)

1. stocks (expect 1 hour for simulation_policy_6)
python stock_mvp/upsert_unique_ticker.py
python stock_mvp/fresh_upsert_subset.py stock_mvp/simulation_policy_5.json





3. Macros (expected 5.3k final series, takes 2 hour total)

python macros/upsert_fresh_categories.py macros/simulation_policy_1.json       
python macros/upsert_fresh_series.py macros/simulation_policy_1.json macros/filters/upsert_categories_blacklist.csv
python macros/upsert_observations.py macros/simulation_policy_1.json

WITH summary AS (
    SELECT 
        sf.series_id,
        sf.frequency_short,
        sf.popularity,
        sf.units_short,
        sf.title,
        MIN(o.date) AS earliest_date,
        MAX(o.date) AS latest_date,
        MIN(o.realtime_start) AS earliest_realtime_start,
        MAX(o.realtime_start) AS latest_realtime_start
    FROM fred_observations AS o
    JOIN fred_series_filtered AS sf 
        ON o.series_id_hash = sf.series_id_hash
    WHERE sf.series_id IN (
        'PPIACO', 'PPOILUSDM', 'PRUBBUSDM', 
        'PSUGAISAUSDM', 'PURANUSDM', 'PWHEAMTUSDM'
    )
    GROUP BY 
        sf.series_id,
        sf.frequency_short,
        sf.popularity,
        sf.units_short,
        sf.title
)
SELECT 
    series_id, 
    earliest_date, 
    latest_date, 
    earliest_realtime_start, 
    latest_realtime_start, 
    popularity, 
    frequency_short, 
    units_short, 
    title 
FROM summary
WHERE earliest_date <= '2015-01-01'
  AND latest_date >= '2026-01-01'
  AND earliest_realtime_start <= '2019-01-01'
  AND latest_realtime_start >= '2026-01-01'
  AND popularity > 10;




