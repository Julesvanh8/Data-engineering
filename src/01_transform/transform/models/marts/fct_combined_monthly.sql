{{
    config(
        materialized='table'
    )
}}

/*
Final fact table combining S&P 500, unemployment, and tax revenue data.
This is the primary analytics-ready table for analysis and dashboarding.
*/

with stock_lags as (
    select * from {{ ref('int_lag_features') }}
),

unemployment_data as (
    select * from {{ ref('stg_unemployment') }}
),

tax_data as (
    select * from {{ ref('int_tax_monthly') }}
)

select
    s.date,
    
    -- S&P 500 metrics
    s.close as sp500_close,
    s.monthly_return as sp500_return,
    s.downturn,
    
    -- Unemployment
    u.unemployment_rate,
    
    -- Tax revenue
    t.tax_revenue_monthly as federal_tax_revenue,
    
    -- Lag features
    s.downturn_lag_1m,
    s.downturn_lag_2m,
    s.downturn_lag_3m,
    s.downturn_lag_4m,
    s.downturn_lag_5m,
    s.downturn_lag_6m,
    s.downturn_lag_7m,
    s.downturn_lag_8m,
    s.downturn_lag_9m,
    s.downturn_lag_10m,
    s.downturn_lag_11m,
    s.downturn_lag_12m

from stock_lags s
left join unemployment_data u on s.date = u.date
left join tax_data t on s.date = t.date

-- Only include rows where we have unemployment data
where u.unemployment_rate is not null

order by s.date
