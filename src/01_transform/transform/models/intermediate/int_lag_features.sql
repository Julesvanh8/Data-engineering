{{
    config(
        materialized='table'
    )
}}

/*
Create lag features for downturn indicators.
Generates 1-12 month lags to analyze delayed effects.
*/

with returns_data as (
    select * from {{ ref('int_monthly_returns') }}
)

select
    date,
    close,
    monthly_return,
    downturn,
    
    -- Lag features: 1-12 months
    lag(downturn, 1) over (order by date) as downturn_lag_1m,
    lag(downturn, 2) over (order by date) as downturn_lag_2m,
    lag(downturn, 3) over (order by date) as downturn_lag_3m,
    lag(downturn, 4) over (order by date) as downturn_lag_4m,
    lag(downturn, 5) over (order by date) as downturn_lag_5m,
    lag(downturn, 6) over (order by date) as downturn_lag_6m,
    lag(downturn, 7) over (order by date) as downturn_lag_7m,
    lag(downturn, 8) over (order by date) as downturn_lag_8m,
    lag(downturn, 9) over (order by date) as downturn_lag_9m,
    lag(downturn, 10) over (order by date) as downturn_lag_10m,
    lag(downturn, 11) over (order by date) as downturn_lag_11m,
    lag(downturn, 12) over (order by date) as downturn_lag_12m
    
from returns_data
order by date
