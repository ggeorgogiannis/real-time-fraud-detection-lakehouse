with expected as (
    select
        cast(tx_datetime as date) as transaction_date,
        count(*)::bigint as transaction_count,
        sum(tx_fraud)::bigint as fraud_count
    from {{ ref("stg_transactions") }}
    group by transaction_date
),

actual as (
    select
        transaction_date,
        transaction_count,
        fraud_count
    from {{ ref("fct_daily_fraud") }}
)

select
    coalesce(actual.transaction_date, expected.transaction_date) as transaction_date,
    actual.transaction_count as actual_transaction_count,
    expected.transaction_count as expected_transaction_count,
    actual.fraud_count as actual_fraud_count,
    expected.fraud_count as expected_fraud_count
from actual
full outer join expected
    on actual.transaction_date = expected.transaction_date
where actual.transaction_count is distinct from expected.transaction_count
   or actual.fraud_count is distinct from expected.fraud_count