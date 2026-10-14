-- Past NWP forecasts, cleaned: radiation and wind cannot be negative, cloud cover is 0–100 %.
select
    plant_id,
    valid_time,
    lead_day,
    greatest(ghi, 0) as ghi,
    greatest(dni, 0) as dni,
    greatest(dhi, 0) as dhi,
    temp_air,
    least(greatest(cloud_cover, 0), 100) as cloud_cover,
    greatest(wind_speed, 0) as wind_speed
from {{ source('bronze', 'nwp_previous_runs') }}
where model = '{{ var("nwp_model") }}'
