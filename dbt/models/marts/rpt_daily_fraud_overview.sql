with customer_activity as (
    select
        transaction_date,
        cast(count(*) as bigint) as active_customer_count,
        cast(
            sum(case when fraud_count > 0 then 1 else 0 end) as bigint
        ) as customers_with_fraud
    from {{ ref("fct_customer_daily_risk") }}
    group by transaction_date
),

terminal_activity as (
    select
        transaction_date,
        cast(count(*) as bigint) as active_terminal_count,
        cast(
            sum(case when fraud_count > 0 then 1 else 0 end) as bigint
        ) as terminals_with_fraud
    from {{ ref("fct_terminal_daily_risk") }}
    group by transaction_date
)

select
    daily.transaction_date,
    daily.transaction_count,
    daily.total_amount,
    daily.fraud_count,
    daily.fraud_rate,
    customers.active_customer_count,
    customers.customers_with_fraud,
    terminals.active_terminal_count,
    terminals.terminals_with_fraud
from {{ ref("fct_daily_fraud") }} as daily
inner join customer_activity as customers
    on daily.transaction_date = customers.transaction_date
inner join terminal_activity as terminals
    on daily.transaction_date = terminals.transaction_date