with terminal_daily as (
    select
        transaction_date,
        terminal_id,
        transaction_count,
        unique_customers,
        total_amount,
        average_amount,
        fraud_count,
        fraud_rate
    from {{ ref("stg_terminal_daily_summary") }}
),

rolling_metrics as (
    select
        *,
        cast(
            sum(transaction_count) over (
                partition by terminal_id
                order by transaction_date
                range between interval '6 days' preceding and current row
            ) as bigint
        ) as rolling_7d_transaction_count,
        sum(total_amount) over (
            partition by terminal_id
            order by transaction_date
            range between interval '6 days' preceding and current row
        ) as rolling_7d_total_amount,
        cast(
            sum(fraud_count) over (
                partition by terminal_id
                order by transaction_date
                range between interval '6 days' preceding and current row
            ) as bigint
        ) as rolling_7d_fraud_count
    from terminal_daily
)

select
    transaction_date,
    terminal_id,
    transaction_count,
    unique_customers,
    total_amount,
    average_amount,
    fraud_count,
    fraud_rate,
    rolling_7d_transaction_count,
    rolling_7d_total_amount,
    rolling_7d_fraud_count,
    cast(rolling_7d_fraud_count as double)
        / nullif(rolling_7d_transaction_count, 0) as rolling_7d_fraud_rate
from rolling_metrics