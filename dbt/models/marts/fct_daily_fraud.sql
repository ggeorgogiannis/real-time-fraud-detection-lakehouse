select
    cast(tx_datetime as date) as transaction_date,
    count(*)::bigint as transaction_count,
    sum(tx_amount) as total_amount,
    sum(tx_fraud)::bigint as fraud_count,
    avg(tx_fraud) as fraud_rate
from {{ ref("stg_transactions") }}
group by transaction_date