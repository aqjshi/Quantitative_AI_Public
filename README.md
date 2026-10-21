## Implicit Environment Setup


1. **Setup python and IDE on Machine**
2. **Setup PgAdmin4, with the SysAdmin Permissions.**
3. **Acquire a .env from an admin. Keep a copy of your .env, swap it in to replace mine.**

### Admin Setup (ADMIN ONLY)

``` powershell
pip freeze | ForEach-Object { ($_ -split '==')[0] } > requirements.txt
``` 



### Engineer Setup


if windows:
``` powershell
python -m venv venv
./venv/scripts/activate.ps1
pip install -r requirements.txt
```


else:
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

# Pull

```
git checkout main
git pull origin main 
```


# Commit
Engineers committing needs to checkout a new branch with one of the following prefixes. Modify the below code such that it matches the purpose of the commit. Do not push to main branch. 


1. **feat/**
2. **fix/**
3. **refactor/**
4. **docs/**
5. **chore/**
6. **test/**



```
git checkout -b {valid-prefix}/{hyphen-separated-intent}
git add .
git commit -m "feat: implement stripe webhook handler"
git push -u origin {valid-prefix}/{hyphen-separated-intent}
```

If Engineer has conflict in the type of: 

[rejected] - non-fast-forward
error: failed to push some refs to '...'
hint: Updates were rejected because the remote contains work that you do not have locally.

Run 
```
git pull --rebase origin main
```


# Merge
Admin will resolve conflict on github directly, as soon as possible, wait for admin to approve of merge on separate branch. 
```
git checkout main
git pull origin main
git branch -d {valid-prefix}/{hyphen-separated-intent}
git fetch --prune
```



# Cleaning Corrupted Database
Cleaning The Database:
```sql
SELECT table_schema, table_name 
FROM information_schema.tables 
WHERE table_type = 'BASE TABLE'
ORDER BY table_schema, table_name;
```


**PRUNE PUBLIC TABLES ONLY**

Checking size of table 
```sql
SELECT pg_size_pretty(pg_total_relation_size('{table_name}'));
```

If a table is messed up, you have to regularly prune out corrupted table. Contact admin for assistance. 



# Approved Scripts
## 1. Stocks Precomputation and Universe Seeding (run once) 
(expect 1 hour for simulation_policy_6)

``` powershell
python stock_mvp/upsert_unique_ticker.py
```


``` powershell
python stock_mvp/fresh_upsert_subset.py stock_mvp/simulation_policy_5.json
```
### Maintaining Database
This is a quick patch to prime the database with most recent observations for the subset of data that is immediately required. This is not a substitute for seeding the database. 
``` powershell
python stock_mvp/maintain_subset.py stock_mvp/simulation_policy_4.json  
``` 





## 2. Options 
(expect 6 hours for simulation_policy_5)
``` powershell
python stock_mvp/fresh_upsert_options.py stock_mvp/simulation_policy_5.json
``` 


## 3. Macros (expected 5.3k final series, takes 2 hour total)
``` powershell
python macros/upsert_fresh_categories.py macros/simulation_policy_1.json      
```


``` powershell 
python macros/upsert_fresh_series.py macros/simulation_policy_1.json macros/filters/upsert_categories_blacklist.csv
```


``` powershell

python macros/upsert_observations.py macros/simulation_policy_1.json
```


``` sql
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
```
### Maintaining Database
This is a quick patch to prime the database with most recent observations for the subset of data that is immediately required. This is not a substitute for seeding the database. 
``` powershell
python macros/maintain_observations.py macros/simulation_policy_1.json macros/exogeneity/seed_3.txt
``` 


## 4. Forex
Waiting on admin to publish.
## 5. Commodities 
Under Construction.
## 6. Bonds
Under Construction.
## 7. Indices

``` powershell
python indices/fresh_upsert_etf.py indices/master_policy.json
```
``` powershell
python indices/maintain_etf.py indices/master_policy.json    
```
### DEPRECATED: DO NOT USE
``` powershell
python indices/fresh_upsert_subset.py indices/master_policy.json
``` 

## 8. News
Waiting on admin to publish.
## 9. Congress Policy
Waiting on admin to publish.
## 10. Real Estate
Waiting on admin to publish.

## 11. Portfolio Design 
Waiting on admin to publish.

## 12. Pricing Simulations

(Nowcasting simulation takes 10 minutes 5 year, 27 ish variable)
``` powershell

python macros/async_validation.py macros/simulation_policy_1.json
``` 

(takes around 30 minutes 6 years with stride 1 week, 150 tickers.)
``` powershell
python pricing/validation.py pricing/simulation_policy_bigger.json
``` 
``` powershell
python pricing/async_validation.py pricing/simulation_policy_bigger.json pricing/seed_3.txt
```
### Asynchronous Operation and Health Check
#### Pricing
``` powershell
python stock_mvp/async/async_f2.py pricing/simulation_policy_small.json   
``` 
``` sql
WITH e AS (SELECT DISTINCT figi_group FROM async_f2 WHERE ticker IN ('FB','META'))
SELECT (to_timestamp(stamp/1e9) AT TIME ZONE 'UTC')::date, ticker,
       derived_market_value, market_cap,
       market_cap/NULLIF(derived_market_value,0),
       CASE WHEN ticker='FB'   THEN derived_market_value END,
       CASE WHEN ticker='META' THEN derived_market_value END
FROM async_f2 WHERE figi_group IN (SELECT figi_group FROM e)
ORDER BY 1;
``` 


#### Short Interest
``` powershell
python stock_mvp/async_short_interest.py pricing/simulation_policy_small.json
``` 

``` sql
SELECT
    TO_TIMESTAMP(stamp / 1e9)::date AS day,
    ROUND(AVG(pct_short_of_market_cap)
          FILTER (WHERE TO_TIMESTAMP(stamp / 1e9) < '2022-07-01'), 4) AS pct_as_fb,
    ROUND(AVG(pct_short_of_market_cap)
          FILTER (WHERE TO_TIMESTAMP(stamp / 1e9) >= '2022-07-01'), 4) AS pct_as_meta,
    ROUND(AVG(pct_short_of_market_cap), 4)   AS pct_combined,
    ROUND(AVG(adj_price), 2)                 AS avg_price,
    ROUND(AVG(market_cap) / 1e9, 1)          AS market_cap_bn
FROM async_short_interest
WHERE figi_group = (SELECT figi_group FROM async_short_interest WHERE ticker = 'META' LIMIT 1)
GROUP BY TO_TIMESTAMP(stamp / 1e9)::date
ORDER BY day ASC;
``` 



















APPENDIX (IGNORE)
short interest filter demo 
``` powershell
python pricing/demo/si_filter.py pricing/simulation_policy_bigger.json
``` 

*** modify exogenou/seed.txt, then update the config file to register your intended seed ****  
``` powershell
python macros/ADF.py macros/simulation_policy_1.json 
``` 

``` powershell
python macros/validation.py macros/simulation_policy_1.json
``` 


Benchmarking 
``` powershell
python pricing/portfolio.py pricing/simulation_policy_small.json
``` 


