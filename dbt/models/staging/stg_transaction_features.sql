select *
from {{ source("lakehouse", "gold_transaction_features") }}