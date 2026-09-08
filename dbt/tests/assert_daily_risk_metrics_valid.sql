select
    'customer' as entity_type,
    transaction_date,
    customer_id as entity_id
from {{ ref("fct_customer_daily_risk") }}
where
    transaction_count <= 0
    or total_amount < 0
    or fraud_count < 0
    or fraud_count > transaction_count
    or fraud_rate < 0
    or fraud_rate > 1
    or rolling_7d_transaction_count < transaction_count
    or rolling_7d_total_amount < 0
    or rolling_7d_fraud_count < fraud_count
    or rolling_7d_fraud_count > rolling_7d_transaction_count
    or rolling_7d_fraud_rate < 0
    or rolling_7d_fraud_rate > 1

union all

select
    'terminal' as entity_type,
    transaction_date,
    terminal_id as entity_id
from {{ ref("fct_terminal_daily_risk") }}
where
    transaction_count <= 0
    or total_amount < 0
    or fraud_count < 0
    or fraud_count > transaction_count
    or fraud_rate < 0
    or fraud_rate > 1
    or rolling_7d_transaction_count < transaction_count
    or rolling_7d_total_amount < 0
    or rolling_7d_fraud_count < fraud_count
    or rolling_7d_fraud_count > rolling_7d_transaction_count
    or rolling_7d_fraud_rate < 0
    or rolling_7d_fraud_rate > 1