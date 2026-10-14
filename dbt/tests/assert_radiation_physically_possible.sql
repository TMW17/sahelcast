-- Fails if a forecast or observation shows more sunshine than can reach the ground.
-- 1,400 W/m² on a horizontal surface is above anything observed at Algerian latitudes.
select plant_id, valid_time, obs_ghi, nwp2_ghi
from {{ ref('fct_hourly') }}
where obs_ghi > 1400 or nwp2_ghi > 1400 or obs_ghi < 0 or nwp2_ghi < 0
