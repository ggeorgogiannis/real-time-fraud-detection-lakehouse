select
    transaction_date,
    customer_id,
    count(*) as row_count
from {{ ref("stg_customer_daily_summary") }}
group by transaction_date, customer_id
having count(*) > 1