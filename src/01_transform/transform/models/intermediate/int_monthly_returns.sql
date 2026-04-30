{{
    config(
        materialized='table'
    )
}}

/*
Calculate monthly returns and downturn indicators for S&P 500.
A downturn is defined as a monthly return <= -5%.
*/

with sp500_data as (
    select * from {{ ref('stg_sp500') }}
)

select
    date,
    sp500_close as close,
    sp500_pct_change / 100.0 as monthly_return,
    case
        when sp500_pct_change / 100.0 <= -0.05 then 1
        else 0
    end as downturn
from sp500_data
where sp500_pct_change is not null
order by date
