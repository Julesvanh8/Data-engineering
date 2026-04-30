{{
    config(
        materialized='table'
    )
}}

/*
Staging model for unemployment (UNRATE) data from SQLite.
Standardizes date format and column names.
*/

select
    date(date) as date,
    rate as unemployment_rate
from unemployment
order by date
