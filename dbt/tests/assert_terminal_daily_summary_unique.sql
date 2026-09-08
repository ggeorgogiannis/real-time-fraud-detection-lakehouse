select
    transaction_date,
    terminal_id,
    count(*) as row_count
from {{ ref("stg_terminal_daily_summary") }}
group by transaction_date, terminal_id
having count(*) > 1