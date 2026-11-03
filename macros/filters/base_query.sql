WITH CategorizedSeries AS (
    SELECT 
        id, category_id, depth, series_id, series_id_hash, realtime_start, realtime_end,  
        title, observation_start, observation_end, frequency, frequency_short,  
        units, units_short, seasonal_adjustment, seasonal_adjustment_short, 
        last_updated, popularity, group_popularity, notes,


        -- 2. Deduplication Policy Window Function
        ROW_NUMBER() OVER (
            PARTITION BY title 
            ORDER BY 
                CASE UPPER(frequency_short)
                    WHEN 'D' THEN 1  -- Daily
                    WHEN 'W' THEN 2  -- Weekly
                    WHEN 'M' THEN 3  -- Monthly
                    WHEN 'Q' THEN 4  -- Quarterly
                    ELSE 5           -- Fallback
                END ASC,
                popularity DESC,
                id ASC
        ) AS title_rank

    FROM fred_series_unfiltered
    WHERE 
        -- Whitelisted series bypass all basic WHERE clause filters
        series_id IN            ('ICNSA',
                                'CCNSA',
                                'CPIAUCNS',
                                'DRTSCILM',
                                'TLBACBW027NBOG',
                                'TOTBKCRNSA',
                                'MRTSSM7225USN', 
                                'INDPRO', 
                                'PCE', 
                                'TCU', 
                                'IQ', 
                                'MORTGAGE30US'
                                ) 
        OR (
            -- Target Raw Unadjusted Values
            seasonal_adjustment_short ILIKE 'NSA'
            AND UPPER(frequency_short) NOT IN ('A')
            
            -- Date Range / Data Density Guardrails
            AND observation_start <= '2010-01-01'
            AND observation_end >= '2025-09-01'

            -- Units & Transformations Exclusions
            AND units_short NOT ILIKE '+1 or 0'
            AND units_short NOT ILIKE '%Number%'
            AND units NOT ILIKE '%Percent Change%'
            AND units NOT ILIKE '%Growth Rate%'
            AND units NOT ILIKE '%annual%'

            -- Master Direct Blacklist (Clean ILIKE Exclusions)
            AND title NOT ILIKE '%discontinued%'
            AND title NOT ILIKE '%population%'
            AND title NOT ILIKE '%Race%'
            AND title NOT ILIKE '% age %'
            AND title NOT ILIKE '%Yrs.%'
            AND title NOT ILIKE '%percentile%'
            AND title NOT ILIKE '%veteran%'
            AND title NOT ILIKE '%disab%'
            AND title NOT ILIKE '%white%'
            AND title NOT ILIKE '%black%'
            AND title NOT ILIKE '%asian%'
            AND title NOT ILIKE '%hisp%'
            AND title NOT ILIKE '%gradu%'
            AND title NOT ILIKE '%college%'
            AND title NOT ILIKE '%school%'
            AND title NOT ILIKE '%degree%'
            AND title NOT ILIKE '%Financial Report%'
            AND title NOT ILIKE '%State Tax Collections%'
            AND title NOT ILIKE '%Applications%'
            AND title NOT ILIKE '%Foreign%'
            AND title NOT ILIKE '%Spliced%'
            AND title NOT ILIKE '%Expenses%'
            AND title NOT ILIKE '%Other %'
            AND title NOT ILIKE '%Banks%'
            AND title NOT ILIKE '%local%'
            AND title NOT ILIKE '%from Business%'
            AND title NOT ILIKE '%from Governments%' 
            AND title NOT ILIKE '%Formations%' 
            AND title NOT ILIKE '%Harmonized%' 
            AND title NOT ILIKE '%Total Revenue%' 
            AND title NOT ILIKE '%Total Value of Issues%' 
            AND title NOT ILIKE '%TSSOS%' 
            AND title NOT ILIKE '%State%'
            AND title NOT ILIKE '%Chained%'
            AND title NOT ILIKE '%End Use%'
            AND title NOT ILIKE '%average%'
            AND title NOT ILIKE '%Flow of Funds%'
            AND title NOT ILIKE '%FDIC%'
            AND title NOT ILIKE '%Federal Debt Held by%'
            AND title NOT ILIKE '%Wealth Percentiles%'
        )
)
SELECT 
    id, category_id, depth, series_id, series_id_hash, realtime_start, realtime_end,  
    title, observation_start, observation_end, frequency, frequency_short,  
    units, units_short, seasonal_adjustment, seasonal_adjustment_short, 
    last_updated, popularity, group_popularity, notes
FROM CategorizedSeries
WHERE 
    -- Retain rank 1 OR any Whitelisted series
    title_rank = 1 
ORDER BY popularity DESC;