with expected as (
    select
        transaction_date,
        cast(count(*) as bigint) as transaction_count,
        sum(tx_amount) as total_amount,
        cast(sum(tx_fraud) as bigint) as fraud_count
    from {{ ref("stg_transaction_features") }}
    group by transaction_date
),

actual as (
    select
        transaction_date,
        transaction_count,
        total_amount,
        fraud_count
    from {{ ref("fct_daily_fraud") }}
)

select
    coalesce(expected.transaction_date, actual.transaction_date) as transaction_date
from expected
full outer join actual
    on expected.transaction_date = actual.transaction_date
where
    expected.transaction_date is null
    or actual.transaction_date is null
    or expected.transaction_count is distinct from actual.transaction_count
    or expected.fraud_count is distinct from actual.fraud_count
    or abs(expected.total_amount - actual.total_amount) > 0.000001