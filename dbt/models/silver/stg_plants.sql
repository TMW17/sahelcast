-- The plant registry plus the ground elevation that Open-Meteo reported for each site.
select
    p.plant_id,
    p.name,
    p.country,
    p.wilaya,
    p.latitude,
    p.longitude,
    p.capacity_mw,
    p.capacity_basis,
    p.ac_mw,
    p.dc_mw,
    p.tilt_deg,
    p.azimuth_deg,
    coalesce(e.elevation_m, 0) as elevation_m,
    p.status,
    p.coord_quality
from {{ source('bronze', 'plants') }} as p
left join {{ source('bronze', 'site_elevation') }} as e using (plant_id)
