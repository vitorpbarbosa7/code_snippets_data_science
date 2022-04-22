select
    user_id
    , count(*) as num_transactions
    , sum(amount) as total_amount
from
    transactions
where
    user_id = {{ ref }}
    and transaction_date = {{ particao_publico }}
group by
    user_id