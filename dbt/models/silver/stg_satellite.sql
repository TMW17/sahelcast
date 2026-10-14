-- Satellite irradiance with a quality flag. These are coarse "physically possible" limits;
-- rows that fail keep their values in bronze but are not used as observations.
select
    plant_id,
    valid_time,
    ghi,
    dni,
    dhi,
    coalesce(
        ghi between 0 and 1400
        and dni between 0 and 1200
        and dhi between 0 and 900
        and dhi <= ghi + 50,
        false
    ) as qc_ok
from {{ source('bronze', 'satellite') }}
where model = '{{ var("satellite_model") }}'
