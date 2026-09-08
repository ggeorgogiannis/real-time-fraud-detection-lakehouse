select *
from {{ ref("fct_daily_fraud") }}
where fraud_rate < 0
   or fraud_rate > 1