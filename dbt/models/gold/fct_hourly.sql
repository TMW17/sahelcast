-- One row per plant and hour: the satellite "observations", reanalysis temperature and wind,
-- and the NWP forecasts that were made about 1 and 2 days before that hour (nwp1_*, nwp2_*).
{%- set leads = var("lead_days") %}
{%- set cols = ["ghi", "dni", "dhi", "temp_air", "cloud_cover", "wind_speed"] %}

with nwp as (
    select
        plant_id,
        valid_time
        {%- for n in leads %}
        {%- for c in cols %},
        max({{ c }}) filter (where lead_day = {{ n }}) as nwp{{ n }}_{{ c }}
        {%- endfor %}
        {%- endfor %}
    from {{ ref('stg_nwp') }}
    group by plant_id, valid_time
),

spine as (
    select plant_id, valid_time from nwp
    union
    select plant_id, valid_time from {{ ref('stg_satellite') }}
)

select
    spine.plant_id,
    spine.valid_time,
    case when sat.qc_ok then sat.ghi end as obs_ghi,
    case when sat.qc_ok then sat.dni end as obs_dni,
    case when sat.qc_ok then sat.dhi end as obs_dhi,
    rea.temp_air as obs_temp_air,
    rea.wind_speed as obs_wind_speed,
    nwp.* exclude (plant_id, valid_time)
from spine
left join {{ ref('stg_satellite') }} as sat using (plant_id, valid_time)
left join {{ ref('stg_reanalysis') }} as rea using (plant_id, valid_time)
left join nwp using (plant_id, valid_time)
