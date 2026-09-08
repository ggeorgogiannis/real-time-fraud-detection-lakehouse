select
    transaction_date,
    customer_id,
    count(*) as row_count
from {{ ref("fct_customer_daily_risk") }}
group by transaction_date, customer_id
having count(*) > 1