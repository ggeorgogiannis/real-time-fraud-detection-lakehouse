select *
from {{ source("lakehouse", "silver_transactions") }}