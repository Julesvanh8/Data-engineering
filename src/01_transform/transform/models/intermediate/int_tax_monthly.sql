{{
    config(
        materialized='table'
    )
}}

/*
Convert quarterly tax revenue (SAAR) to monthly estimates.
Uses forward-fill logic to spread quarterly values across 3 months.
Divides SAAR by 12 to get monthly rate.
*/

with tax_quarterly as (
    select * from {{ ref('stg_tax_revenue') }}
),

-- Generate all months between min and max date
date_spine as (
    select distinct date as month_date
    from {{ ref('stg_sp500') }}
    where date >= (select min(date) from tax_quarterly)
      and date <= (select max(date) from tax_quarterly)
),

-- Join quarterly data with monthly spine and get most recent quarter for each month
monthly_base as (
    select
        s.month_date,
        t.date as quarter_date,
        t.tax_revenue_quarterly_saar,
        row_number() over (
            partition by s.month_date 
            order by t.date desc
        ) as rn
    from date_spine s
    left join tax_quarterly t
        on t.date <= s.month_date
)

select
    month_date as date,
    quarter_date,
    tax_revenue_quarterly_saar,
    -- Convert SAAR to monthly: divide by 12
    tax_revenue_quarterly_saar / 12.0 as tax_revenue_monthly
from monthly_base
where rn = 1
order by month_date
