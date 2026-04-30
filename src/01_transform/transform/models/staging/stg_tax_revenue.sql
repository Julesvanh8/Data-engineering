{{
    config(
        materialized='table'
    )
}}

/*
Staging model for federal income tax revenue from SQLite.
Standardizes date format and column names.
Data is quarterly SAAR (Seasonally Adjusted Annual Rate).
*/

select
    date(date) as date,
    receipts_bn as tax_revenue_quarterly_saar,
    source as data_source
from tax_revenue
where source = 'FRED_quarterly'
order by date
