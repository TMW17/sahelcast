-- Fails if any plant has two rows for the same hour (the table's grain is plant x hour).
select plant_id, valid_time, count(*) as n
from {{ ref('fct_hourly') }}
group by plant_id, valid_time
having count(*) > 1
