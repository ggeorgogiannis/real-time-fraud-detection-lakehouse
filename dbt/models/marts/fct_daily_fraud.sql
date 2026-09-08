select
    transaction_date,
    cast(count(*) as bigint) as transaction_count,
    sum(tx_amount) as total_amount,
    cast(sum(tx_fraud) as bigint) as fraud_count,
    avg(tx_fraud) as fraud_rate
from {{ ref("stg_transaction_features") }}
group by transaction_date