select *
from {{ ref("rpt_daily_fraud_overview") }}
where
    transaction_count <= 0
    or total_amount < 0
    or fraud_count < 0
    or fraud_count > transaction_count
    or fraud_rate < 0
    or fraud_rate > 1
    or active_customer_count <= 0
    or customers_with_fraud < 0
    or customers_with_fraud > active_customer_count
    or active_terminal_count <= 0
    or terminals_with_fraud < 0
    or terminals_with_fraud > active_terminal_count
    or (fraud_count = 0 and customers_with_fraud <> 0)
    or (fraud_count = 0 and terminals_with_fraud <> 0)
    or (fraud_count > 0 and customers_with_fraud = 0)
    or (fraud_count > 0 and terminals_with_fraud = 0)