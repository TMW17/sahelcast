-- Air temperature and wind speed, used by the PV model to estimate module temperature.
select
    plant_id,
    valid_time,
    temp_air,
    greatest(wind_speed, 0) as wind_speed
from {{ source('bronze', 'reanalysis') }}
where model = '{{ var("reanalysis_model") }}'
