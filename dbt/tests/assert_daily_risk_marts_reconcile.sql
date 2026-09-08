with daily_fraud as (
    select
        transaction_date,
        transaction_count,
        total_amount,
        fraud_count
    from {{ ref("fct_daily_fraud") }}
),

customer_totals as (
    select
        transaction_date,
        sum(transaction_count) as transaction_count,
        sum(total_amount) as total_amount,
        sum(fraud_count) as fraud_count
    from {{ ref("fct_customer_daily_risk") }}
    group by transaction_date
),

terminal_totals as (
    select
        transaction_date,
        sum(transaction_count) as transaction_count,
        sum(total_amount) as total_amount,
        sum(fraud_count) as fraud_count
    from {{ ref("fct_terminal_daily_risk") }}
    group by transaction_date
)

select
    coalesce(
        daily.transaction_date,
        customer.transaction_date,
        terminal.transaction_date
    ) as transaction_date
from daily_fraud as daily
full outer join customer_totals as customer
    on daily.transaction_date = customer.transaction_date
full outer join terminal_totals as terminal
    on coalesce(daily.transaction_date, customer.transaction_date)
        = terminal.transaction_date
where
    daily.transaction_date is null
    or customer.transaction_date is null
    or terminal.transaction_date is null
    or daily.transaction_count is distinct from customer.transaction_count
    or daily.transaction_count is distinct from terminal.transaction_count
    or daily.fraud_count is distinct from customer.fraud_count
    or daily.fraud_count is distinct from terminal.fraud_count
    or abs(daily.total_amount - customer.total_amount) > 0.000001
    or abs(daily.total_amount - terminal.total_amount) > 0.000001