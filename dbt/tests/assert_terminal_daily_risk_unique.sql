select
    transaction_date,
    terminal_id,
    count(*) as row_count
from {{ ref("fct_terminal_daily_risk") }}
group by transaction_date, terminal_id
having count(*) > 1