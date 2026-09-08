select *
from {{ source("lakehouse", "gold_customer_daily_summary") }}