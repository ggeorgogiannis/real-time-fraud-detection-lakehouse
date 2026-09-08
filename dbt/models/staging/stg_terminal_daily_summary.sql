select *
from {{ source("lakehouse", "gold_terminal_daily_summary") }}