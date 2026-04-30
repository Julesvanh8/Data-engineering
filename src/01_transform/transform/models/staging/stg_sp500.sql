{{
    config(
        materialized='table'
    )
}}

/*
Staging model for S&P 500 data from SQLite.
Standardizes date format and column names.
*/

select
    date(date) as date,
    close as sp500_close,
    pct_change as sp500_pct_change
from sp500
order by date
