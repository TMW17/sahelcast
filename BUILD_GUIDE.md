# SahelCast: step-by-step build guide

**From an empty folder to a finished portfolio project: day-ahead solar forecasts for Algerian
plants, with honest uncertainty ranges, used to plan a battery.**

---

## What you are building, in plain words

Every day at 12:00 UTC, SahelCast answers one question for tomorrow: *how much electricity will
Algeria's solar plants produce, hour by hour, and how sure are we?*

1. It **collects data**: weather forecasts, satellite measurements of sunshine, and temperature,
   for each plant (all free, from Open-Meteo). It stores them in a small lakehouse on your laptop.
2. It **turns weather into power**: a physics library (pvlib) computes what each plant produces
   under that weather.
3. It **forecasts tomorrow** as a range, not one number ("between 180 and 240 MW at noon, most
   likely 215"), using four models from simple to state-of-the-art.
4. It **makes the ranges honest**: if it says "80% sure", the truth must fall inside 80% of the
   time. A technique called conformal prediction checks and fixes this every day.
5. It **makes decisions**: each plant promises tomorrow's delivery to the grid and uses a
   battery to keep that promise. Better forecasts mean fewer broken promises, and that is
   measured in money.
6. It **runs by itself and shows its results**: a scheduler (Dagster), an API (FastAPI) and a
   free website (GitHub Pages).

## What you will have at the end

- A public GitHub repository with tested, documented code and CI.
- A lakehouse (DuckLake) with bronze, silver and gold layers, data tests (dbt) and time travel.
- A leak-free backtest of four models, with calibrated uncertainty and statistical tests.
- A decision study showing what the forecasts are worth.
- A daily forecast, a REST API and a public website.
- A short technical report and CV bullets based on your own measured results.

**Time:** about 12 weeks part-time (the minimum CV-worthy version takes about 7 weeks).
**Cost:** $0. Everything runs on your Mac with free data and free hosting.

## How to use this guide and the starter repository

You received a **starter repository** (`sahelcast-starter.zip`). It contains working code for
every step below, 27 automated tests, and an offline demo that runs the whole pipeline on
synthetic data. This guide explains each part, tells you exactly what to run, and gives you the
parts to build yourself ("Your turn").

Recommended way to work:

1. Follow the phases in order. For each step, **read the file first**, run it, and check the
   result before moving on. Do not skip the checks.
2. Commit to git at the end of each step with a clear message, so your history shows your work.
3. Do the "Your turn" tasks. They are what turns the starter into *your* project: real data,
   real validation, your own analysis.

> **Make it yours.** Interviewers will ask about any line in your repository. Change what you
> disagree with, rename what you find unclear, and be honest about which tools helped you.

## The whole pipeline on one page

```text
                    ┌────────────── Open-Meteo (free, no key) ──────────────┐
                    │ Previous Runs API   Satellite API   Historical API    │
                    │ (past forecasts)    (sunshine seen  (temperature,     │
                    │                      from space)     wind)            │
                    └───────────────────────────┬───────────────────────────┘
Phase 1                                         ▼
  config/plants.csv ──► ingest.py ──► BRONZE  (DuckLake: raw, typed, every version kept)
                                        │
                                        ▼  dbt build (+ data tests)
                                      SILVER (clean) ──► GOLD fct_hourly (one row per plant-hour)
Phase 2                                                    │
                                                           ▼  physics.py (pvlib)
                                                     GOLD pv_hourly: actual, forecast, clear-sky MW
Phase 3                                                    │
                          dataset.py: plants + wilayas + country, leak-free daily schedule
                                                           │
             ┌──────────────┬──────────────┬───────────────┼───────────────┐
             ▼              ▼              ▼               ▼               │
        persistence     physical       LightGBM        Chronos-2           │
             └──────────────┴──────┬───────┴───────────────┘               │
                                   ▼                                       │
                    conformal.py: calibrate every quantile (honest ranges)  │
                                   ▼                                       │
                    backtest.py + metrics.py: scores, coverage, tests       │
Phase 4                            ▼                                       │
                    reconcile.py: plant + wilaya = national                │
Phase 5                            ▼                                       │
                    dispatch.py: battery plans, profit, regret             │
Phase 6                            ▼                                       ▼
            live.py (tomorrow) ─► publish.py ─► report.py (website), api.py (REST)
                           Dagster runs it every day at 12:15 UTC
```

## Ten words you will meet

| Word | Meaning |
|---|---|
| **GHI, DNI, DHI** | Sunshine on flat ground (global), from the sun's direction (direct), and from the sky (diffuse), in W/m² |
| **NWP** | Numerical weather prediction: a weather model's forecast (here ECMWF IFS) |
| **Lead day** | How far ahead a forecast was made. `previous_day2` = made about 48 hours before |
| **Clear-sky** | What a plant would produce under a perfectly clear sky |
| **Capacity factor (CF)** | Power divided by the plant's capacity: 0 = nothing, 1 = full power |
| **Quantile** | q10 = "90% chance the truth is above this"; q50 = the median; q90 = "90% chance below" |
| **Coverage** | How often the truth fell inside a range. An 80% range should cover 80% |
| **Lakehouse** | Data files (Parquet) plus a catalog that gives database features (versions, transactions) |
| **Backtest** | Re-running the forecast for every past day, using only what was known on that day |
| **Regret** | Profit lost compared with knowing the future. Lower is better |

---

# Phase 0: Set up your Mac (1–2 days)

## Step 0.1: Install the tools

**Goal:** a working Python environment and the command-line tools.

Open **Terminal** and run:

```bash
# Homebrew, the Mac package manager (skip if `brew --version` already works)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# uv (Python and packages), git, OpenMP (needed by LightGBM on macOS), GitHub CLI
brew install uv git libomp gh

# A Python version for the project
uv python install 3.12
```

**Check:** `uv --version` and `git --version` both print a version.

## Step 0.2: Get the starter running

```bash
mkdir -p ~/projects && cd ~/projects
unzip ~/Downloads/sahelcast-starter.zip      # creates the folder sahelcast/
cd sahelcast
git init
make setup                                    # installs everything, including Chronos-2 and PyTorch
make test                                     # 27 tests, about 2 minutes
```

`make setup` runs `uv sync --all-extras`, which creates `.venv/` from `uv.lock` (the exact package
versions that were tested), then installs the git hooks.

**Check:** the last line says `27 passed`. The first run downloads the DuckLake extension for
DuckDB, so you need internet.

**Watch out:** if LightGBM fails with an error about `libomp.dylib`, run `brew install libomp`.

## Step 0.3: Run the offline demo

```bash
make demo
open data/demo/site/index.html
```

The demo runs the **whole pipeline** (ingest, dbt, pvlib, three models, calibration,
reconciliation, dispatch, website) on **synthetic weather** made by
`src/sahelcast/fake_openmeteo.py`. It takes about four minutes and writes everything under
`data/demo/`, separate from your real data. Add Chronos-2 with `uv run sahelcast demo --with-chronos`
(it downloads the model the first time).

**Check:** the website opens and shows a "Synthetic data" warning banner, a forecast chart, model
scores, calibration dots and dispatch results.

> **Never report the demo's numbers.** They describe invented weather. They only prove the
> pipeline works end to end.

## Step 0.4: Put it on GitHub

```bash
gh auth login                                 # once
git add . && git commit -m "Start SahelCast from the starter repository"
gh repo create sahelcast --public --source=. --remote=origin --push
```

Pushing starts the CI workflow (`.github/workflows/ci.yml`): lint plus all tests on a free
GitHub-hosted runner. Open the **Actions** tab to watch it. CI does not install PyTorch, so it
reports 26 passed and 1 skipped (the Chronos-2 test).

**Check:** a green tick next to your commit on GitHub.

### Phase 0 checkpoint
- [ ] `make test` passes locally, CI is green on GitHub
- [ ] The demo website opens

---

# Phase 1: The data platform (weeks 1–3)

## Step 1.1: The plant registry

**Goal:** know exactly which plants you forecast, and where each fact comes from.

Open `config/plants.csv`. Each row is one plant:

| Column | Meaning |
|---|---|
| `plant_id`, `name`, `wilaya` | Identity; the wilaya is used for the hierarchy (plants add up to wilayas, wilayas to Algeria) |
| `latitude`, `longitude` | Location. `coord_quality` says if it is exact or town-level |
| `capacity_mw`, `capacity_basis` | Size as published: `MWp` (DC panels), `MWac` (grid connection) or plain `MW` (treated as MWp) |
| `tilt_deg`, `azimuth_deg` | Panel angle and direction. Unknown for these plants, so assumed 30° facing south (180°) |
| `status`, `commissioned`, `source` | Where the information comes from |

The starter lists ten plants: eight SKTM plants from 2015–2017 with exact coordinates (Adrar,
Aïn El Ibel I and II, El Kheneg I and II, Oued El Kebrit, Tamanrasset, Telagh, El Abiodh Sidi
Cheikh) from the Global Energy Monitor wiki, plus the 200 MW Tendla plant commissioned in 2026
(town-level coordinates).

`src/sahelcast/config.py` reads this file and derives both capacities: a plant published as
20 MWp gets 20 MW DC and 20 / 1.2 = 16.7 MW AC (the DC/AC ratio is in `settings.yaml`).

**Your turn:** add plants from the Sonelgaz 2025–2026 programme as soon as reliable sources give
their location (El Ghrous in Biskra and others), with the source URL in the `source` column.
Keep `coord_quality` honest.

## Step 1.2: Look at the data in your browser first

**Goal:** see the raw data before writing any code. Paste these URLs into your browser (they are
for El Kheneg, Laghouat). Each returns JSON.

1. **The forecast made 1 and 2 days before** (Previous Runs API):
   ```
   https://previous-runs-api.open-meteo.com/v1/forecast?latitude=33.7298&longitude=2.8164&hourly=shortwave_radiation_previous_day1,shortwave_radiation_previous_day2&models=ecmwf_ifs025&start_date=2024-03-01&end_date=2024-03-02&timezone=GMT
   ```
2. **Sunshine measured from space** (Satellite Radiation API):
   ```
   https://satellite-api.open-meteo.com/v1/archive?latitude=33.7298&longitude=2.8164&hourly=shortwave_radiation&models=eumetsat_sarah3&start_date=2024-03-01&end_date=2024-03-02
   ```
3. **Temperature and wind** (Historical Weather API):
   ```
   https://archive-api.open-meteo.com/v1/archive?latitude=33.7298&longitude=2.8164&hourly=temperature_2m,wind_speed_10m&wind_speed_unit=ms&start_date=2024-03-01&end_date=2024-03-02
   ```
4. **Tomorrow's forecast** (Forecast API):
   ```
   https://api.open-meteo.com/v1/forecast?latitude=33.7298&longitude=2.8164&hourly=shortwave_radiation,temperature_2m&models=ecmwf_ifs025&forecast_days=2&timezone=GMT
   ```

Three facts to understand now, because the whole project depends on them:

- **Radiation is the average of the preceding hour.** The value at `12:00` covers 11:00–12:00 UTC.
- **`previous_day2` is what the model predicted about 48 hours earlier.** At 12:00 UTC today,
  those values already exist for every hour of tomorrow, so a forecast built on them cannot
  secretly use information from the future. This is the key to an honest backtest.
- **The Historical Forecast API is not what it sounds like.** It stitches the first hours of
  each run together, so it is too accurate to stand for a forecast made the day before. Don't
  use it for backtests.

**Check:** in URLs 1 and 2, the values for 1 March 2024 are numbers, not `null`. If the
Previous Runs values are `null`, move `history_start` later.

**Why SARAH-3 for the "observations"?** The Satellite API offers several products. Its
`satellite_radiation_seamless` option picks, for Algeria, a newer product (from the MTG
satellite) that only starts in 2025. SARAH-3, EUMETSAT's climate data record, covers 1983 to
about two days ago with one consistent method, which is what a multi-year backtest needs. The
price is that delay of about two days; Step 3.1 shows how the forecast schedule respects it.

**Rules of the free tier:** non-commercial use (a portfolio qualifies), no API key, fewer than
600 calls per minute, 5,000 per hour and 10,000 per day. A request with more than 10 variables or
more than two weeks counts as several calls. Credit "Weather data by Open-Meteo.com" (CC BY 4.0).

## Step 1.3: The API client

**Goal:** download data politely and never twice.

Read `src/sahelcast/openmeteo.py`. It does four things you will be asked about in interviews:

1. **Rate limit**: waits between requests (`requests_per_minute: 30` in the settings).
2. **Retries** with exponential backoff on network errors and HTTP 429/5xx (the `tenacity` library),
   but not on HTTP 400 (a bad request will not fix itself).
3. **Cache**: every raw response is saved in `data/raw/<endpoint>/<hash>.json`. A second run reads
   the file instead of calling the API. Interrupted downloads resume, and you can rebuild the
   lake offline. Recent days are never cached because providers are still filling them in.
4. **Parsing** into a DataFrame with UTC timestamps and short column names (`ghi`, `dni`, `dhi`,
   `temp_air`, `cloud_cover`, `wind_speed`).

**Check:** `uv run pytest tests/test_openmeteo.py -v` passes (5 tests).

## Step 1.4: The lakehouse

**Goal:** a real lakehouse on your laptop, with versions you can go back to.

Read `src/sahelcast/lake.py`. DuckLake stores table data as Parquet files in `data/lake/files/` and
keeps the catalog (which files make up which table version) in one small database file,
`data/lake/metadata.ducklake`. You get transactions, schema changes and **time travel**.

Loading is an **upsert** with `MERGE`, the same idea as the `INSERT ... ON CONFLICT` upserts in your
PostgreSQL weather pipeline:

```sql
MERGE INTO lake.bronze.satellite AS t USING incoming AS s
  ON t.plant_id IS NOT DISTINCT FROM s.plant_id
 AND t.model    IS NOT DISTINCT FROM s.model
 AND t.valid_time IS NOT DISTINCT FROM s.valid_time
WHEN MATCHED THEN UPDATE SET ghi = s.ghi, dni = s.dni, dhi = s.dhi, ingested_at = s.ingested_at
WHEN NOT MATCHED THEN INSERT BY NAME
```

Loading the same data twice changes nothing. That makes every step safe to re-run.

**One writer at a time.** With a DuckDB file as the catalog, only one process can write. Run steps
one after another, and close any `duckdb` shell before running the pipeline.

## Step 1.5: Download the history

```bash
uv run sahelcast ingest          # from history_start (2024-03-01) to 7 days ago
```

For ten plants and about 2.5 years, that is roughly 900 requests (one per plant, month and source).
At 30 requests per minute, expect **30–90 minutes** the first time. If it stops, run it again: the
cache makes it resume.

Now explore what you loaded. Use the project's own DuckDB (a different DuckDB version could
upgrade or refuse the catalog):

```bash
uv run python
```
```python
from sahelcast.config import load_settings
from sahelcast.lake import Lake

lake = Lake.from_settings(load_settings(), read_only=True)
lake.sql("SELECT count(*) FROM lake.bronze.nwp_previous_runs")       # about 2 rows per plant-hour
lake.sql("""SELECT plant_id, min(valid_time), max(valid_time), count(*)
            FROM lake.bronze.satellite GROUP BY 1 ORDER BY 1""")
lake.sql("SELECT snapshot_id, snapshot_time, changes FROM lake.snapshots() ORDER BY 1 DESC LIMIT 5")
# Time travel: replace 42 with a snapshot id from the list above
lake.sql("SELECT count(*) FROM lake.bronze.satellite AT (VERSION => 42)")
lake.close()
```

`uv run sahelcast snapshots` prints the version history too.

**Check:** every plant has rows for every source, from `history_start` to a week ago. The
number of hours matches the number of days × 24.

**Your turn:** add a freshness check. For each source, compute the latest `valid_time` and fail
if it is more than 3 days old. Put it in a new dbt test (next step) or a pytest.

## Step 1.6: Clean and join with dbt

**Goal:** tested, documented transformations from bronze to gold.

```bash
uv run sahelcast transform
```

This runs `dbt build` in `dbt/` against the lake. It builds five tables and runs eleven tests:

| Model | What it does |
|---|---|
| `silver.stg_plants` | The registry plus the ground elevation Open-Meteo reported (used by pvlib) |
| `silver.stg_nwp` | Past forecasts, cleaned: no negative radiation, cloud cover within 0–100 % |
| `silver.stg_satellite` | Satellite values with a `qc_ok` flag for physically impossible values |
| `silver.stg_reanalysis` | Temperature and wind |
| `gold.fct_hourly` | One row per plant and hour: observations plus `nwp1_*` and `nwp2_*` forecasts |

The tests in `dbt/models/schema.yml` and `dbt/tests/` are your **data quality rules**: unique
plant ids, accepted lead days, one row per plant-hour, radiation within physical limits.

Open `dbt/models/gold/fct_hourly.sql`: a Jinja loop pivots the lead days into columns, so adding
`lead_days: [1, 2, 3]` in the settings adds `nwp3_*` columns without touching the SQL.

To browse the documentation and the lineage graph:

```bash
uv run sahelcast transform docs generate
uv run sahelcast transform docs serve          # opens http://localhost:8080
```

**Check:** `Done. PASS=16 WARN=0 ERROR=0`.

**Why `sahelcast transform` and not `dbt build`?** The wrapper passes the lake paths and the model
names from `settings.yaml`, and runs dbt in its own process (dbt keeps its database connection open
until it exits, which would block the next step).

## Step 1.7 (optional): From batch to streaming

This makes the "streaming lakehouse" part of the project's title literal. Do it after the MVP if you have
time. It needs Docker; on a Mac, use Colima (free and open source) instead of Docker Desktop:

```bash
brew install colima docker docker-compose
colima start --cpu 2 --memory 4
```

Create `compose.yaml`:

```yaml
services:
  redpanda:                 # a Kafka-compatible broker in one container
    image: docker.redpanda.com/redpandadata/redpanda:latest
    command: ["redpanda", "start", "--mode", "dev-container", "--smp", "1",
              "--kafka-addr", "PLAINTEXT://0.0.0.0:9092",
              "--advertise-kafka-addr", "PLAINTEXT://localhost:9092"]
    ports: ["9092:9092"]
  postgres:                 # a catalog that accepts several writers
    image: postgres:16
    environment: {POSTGRES_PASSWORD: sahelcast, POSTGRES_DB: sahelcast}
    ports: ["5432:5432"]
```

Then run `docker-compose up -d` (the Kafka client, `confluent-kafka`, was already installed by
`make setup`) and write two small programs:

- a **producer** that sends every new API response to a topic (`nwp_raw`, `satellite_raw`) with the
  plant id as key, and
- a **consumer** that reads the topics, parses messages with the same `hourly_frame()` function,
  upserts them into bronze, and commits its offset **after** the upsert (so a crash never loses
  data, and the idempotent upsert makes replays harmless).

```python
from confluent_kafka import Consumer

consumer = Consumer({"bootstrap.servers": "localhost:9092", "group.id": "lake-writer",
                     "auto.offset.reset": "earliest", "enable.auto.commit": False})
consumer.subscribe(["satellite_raw"])
while True:
    msg = consumer.poll(1.0)
    if msg is None or msg.error():
        continue
    # parse msg.value() (the JSON payload) -> DataFrame -> lake.upsert("bronze.satellite", ...)
    consumer.commit(msg)
```

With several writers, move the DuckLake catalog to PostgreSQL (you already know it):

```python
con.execute("ATTACH 'ducklake:postgres:dbname=sahelcast host=localhost user=postgres password=sahelcast' "
            "AS lake (DATA_PATH 'data/lake/files/')")
```

> This step was not run while the starter was built. Check the Redpanda flags against the
> Redpanda quickstart for the version you install.

### Phase 1 checkpoint
- [ ] Bronze holds 2+ years for every plant; re-running `ingest` adds nothing new
- [ ] `sahelcast transform` passes all data tests
- [ ] You can query an old snapshot with `AT (VERSION => n)` and explain why that matters
- [ ] Commit: "Phase 1: data platform"

---

# Phase 2: From weather to power (weeks 3–4)

Algerian plants do not publish their hourly output. So SahelCast computes it: satellite
sunshine plus measured temperature go through a physical PV model, and the result is the
"observed" power (sometimes called **virtual telemetry**). The same model applied to the weather
*forecast* gives a physical power forecast. Phase 2 builds this, and checks the method on a real
plant.

## Step 2.1: Where the sun is

Read `solar_geometry()` in `src/sahelcast/physics.py`. For every hour it computes the sun's
position and the clear-sky irradiance (Ineichen model, with pvlib's built-in turbidity data).

One detail matters a lot: Open-Meteo's value at 12:00 is the average of 11:00–12:00, so the sun is
placed at **11:30**. Using 12:00 shifts the whole day by half an hour and creates large errors at
sunrise and sunset. `tests/test_physics.py` checks this.

## Step 2.2: The PV model

Read `pv_ac_mw()`. It follows the PVWatts chain, the standard simple model for a fixed-tilt plant:

1. **Sunshine on the tilted panels** (plane of array) from GHI, DNI and DHI, with the Hay-Davies
   sky model and a desert ground albedo of 0.3.
2. **Cell temperature** from that sunshine, the air temperature and the wind (Faiman model). Hot
   panels produce less: −0.4 % per °C above 25 °C.
3. **DC power** = capacity × sunshine / 1000 W/m² × temperature correction, minus 14 % system
   losses (wiring, soiling, availability).
4. **AC power** through the inverter, capped at the AC capacity (this "clipping" happens on
   clear summer middays when DC capacity is 1.2× AC).

Remember the azimuth convention: pvlib uses **180° = south**; Open-Meteo's own tilted-irradiance
option uses 0° = south. The project only uses pvlib's convention.

## Step 2.3: Build gold.pv_hourly and look at it

```bash
uv run sahelcast physics
```

This writes `gold.pv_hourly` with, for every plant and hour: `pv_actual_mw` (satellite +
reanalysis), `pv_nwp1_mw` and `pv_nwp2_mw` (forecasts made 1 and 2 days ahead), and
`pv_clearsky_mw`.

**Check** with a quick notebook (`uv run --with jupyterlab jupyter lab`):

```python
from sahelcast.config import load_settings
from sahelcast.lake import Lake

lake = Lake.from_settings(load_settings(), read_only=True)
pv = lake.read("gold.pv_hourly", "WHERE plant_id = 'el_kheneg_2'")
day = pv[pv.valid_time.dt.date.astype(str) == "2025-06-21"]
day.plot(x="valid_time", y=["pv_actual_mw", "pv_nwp2_mw", "pv_clearsky_mw"])
```

What you should see: zero at night; the clear-sky curve above the others on most days; a flat
top on clear summer middays (inverter clipping); actual and forecast close on clear days and
different on cloudy or dusty ones.

## Step 2.4 (Your turn): Validate the PV model on a real plant

The physics half of the method can be checked on a US plant with public metered data: NREL PVDAQ
on AWS, free and without an account. **System 4** (Golden, Colorado, 1 kW, fixed 40° tilt facing
south) has plane-of-array irradiance, temperature and AC power.

```bash
brew install awscli
aws s3 ls --no-sign-request s3://oedi-data-lake/pvdaq/csv/                             # metadata files
aws s3 ls --no-sign-request s3://oedi-data-lake/pvdaq/parquet/pvdata/system_id=4/      # the data
```

```python
import pyarrow.dataset as ds
from pyarrow import fs

s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
data = ds.dataset("oedi-data-lake/pvdaq/parquet/pvdata/system_id=4/year=2019/",
                  filesystem=s3, format="parquet").to_table().to_pandas()
# Long format: one row per (timestamp, metric_id, value). Use the metrics metadata file to map
# metric_id to "POA irradiance", "ambient temperature" and "AC power", then pivot to columns.
```

Then run steps 2–4 of the PV chain (cell temperature, PVWatts DC, inverter) on the measured
plane-of-array irradiance, and compare with the measured AC power: daily energy error, hourly
error, and a scatter plot. A good simple model is within a few percent on clear days. Write the
result in your report as "validation of the PV model on metered data".

> Written from the OEDI documentation; the file layout was not checked while the starter was built.
> List the folders first.

## Step 2.5 (optional): Check the satellite against ground truth in the Sahara

Tamanrasset has the only BSRN station in the Sahara (code TAM, 22.79 °N, 5.53 °E, 1,385 m), with
one-minute measurements since 2000. Create a free PANGAEA account, download the TAM files for one
or two years, average them to hours, and compare the measured global irradiance with the
satellite GHI for the same point: bias, error by month, and the worst days (usually dust).

This gives a short, original result: how good satellite irradiance is in the Sahara during dust
events. Respect the licence: cite the dataset, and **never re-publish the raw files** (publish
your download script and your statistics instead). Direct irradiance has been missing since
February 2018 and diffuse is flagged as uncertain, so compare global irradiance only.

### Phase 2 checkpoint
- [ ] `gold.pv_hourly` looks right in plots (night zeros, clipping, clear-sky envelope)
- [ ] The PVDAQ validation notebook gives a number you can quote
- [ ] Commit: "Phase 2: PV physics"

---

# Phase 3: Forecasting (weeks 5–7): the minimum CV-worthy version

## Step 3.1: The rules that prevent cheating

Open `src/sahelcast/dataset.py` and read `make_issues()`. For every past day D, the backtest pretends
it is 12:00 UTC on D:

| Rule | Value | Why |
|---|---|---|
| Forecast issued | 12:00 UTC on day D | A realistic day-ahead deadline |
| Target | the 24 hours of day D+1 | Tomorrow |
| Observations used | up to 00:00 on day D−1 (`obs_cutoff_hours: 36`) | SARAH-3 satellite data are published about two days late |
| Weather covariates | `previous_day2` values | They existed at 12:00 on D for every hour of D+1 |

The calibration (Step 3.7) follows the same rule: it only learns from forecasts whose target day
had ended by the cutoff (`lag_days: 3`). If you change `obs_cutoff_hours`, the settings loader
tells you when `lag_days` must change too.

Now run the most important tests in the project:

```bash
uv run pytest tests/test_no_leakage.py -v
```

Each test **corrupts every observation after the cutoff** and checks that the forecast does not
change at all. If a future bug lets a model peek at tomorrow, these tests fail. In interviews,
this is the answer to "how do you know your backtest is honest?".

## Step 3.2: One panel for plants, wilayas and the country

`build_panel()` builds one table with a row per **node** and hour:

- plants: `DZ/Djelfa/ain_el_ibel_1`
- wilayas (sums of their plants): `DZ/Djelfa`
- the country: `DZ`

Everything is expressed as a **capacity factor** (MW divided by the node's AC capacity), so the same
models can forecast a 12 MW plant and the whole country. Temperature, cloud cover and sun angle are
capacity-weighted averages. An aggregate counts as observed only if **all** its plants are observed.

## Step 3.3: Two baselines every forecast must beat

Read `src/sahelcast/baselines.py`.

**Persistence** ("tomorrow looks like the last two weeks"): for each hour of the day, take the
clear-sky index (actual ÷ clear-sky) observed over the last 14 days, and apply those values to
tomorrow's clear-sky power. The spread of those 14 values gives the quantiles. It is hard to beat
in the Sahara, where most days are clear.

**Physical**: the weather forecast through the PV model (`pv_nwp2_mw`). Its uncertainty comes from
its own recent errors: the quantiles of (actual − forecast) ÷ clear-sky over the last 60 days.

## Step 3.4: LightGBM: learn the weather model's errors

LightGBM does **not** learn power directly. It learns the physical forecast's error:

```
target  r = (actual − physical forecast) ÷ clear-sky capacity factor
forecast  = physical forecast + clear-sky × predicted quantile of r
```

One model per quantile (quantile loss), retrained every 30 days on everything observed before.
Features: the forecast clear-sky index, cloud cover, clear-sky power, sun angle, hour and latitude.

**Why not learn power directly?** The starter tried that first. On the synthetic demo data, a
model trained on winter and spring met July temperatures it had never seen, and tree models cannot
extrapolate: its July forecasts were 0.16 too low (about a third of the output). Physics already models temperature and
sun, so the model should only learn where the weather forecast goes wrong. This hybrid
physics-plus-ML design is a strong interview story: it is a real mistake, found by looking at bias
per month, and fixed with domain knowledge.

## Step 3.5: Chronos-2, a time-series foundation model

Read `src/sahelcast/chronos_model.py`. Chronos-2 (Amazon, 120M parameters, Apache-2.0) forecasts
**zero-shot**: no training on your data. For each node and day it receives the last 28 days of
actual capacity factor, plus two **covariates** known for the past and the future (the physical
forecast and the clear-sky capacity factor), and returns quantiles for every hour up to the end of
tomorrow.

```bash
uv run python scripts/try_chronos.py 2025-06-01     # one week; nothing is written to the lake
```

It prints how long the week took, an estimate for a full year, and the national forecast for the
first day. The first run downloads the model from Hugging Face (about 0.5 GB). On an Apple Silicon
Mac, set `device: mps` in `settings.yaml` to use the GPU (if you see an "operation not supported on
MPS" error, set `PYTORCH_ENABLE_MPS_FALLBACK=1`, or go back to `cpu`). For faster runs, try
`chronos_model_id: autogluon/chronos-2-small` (28M parameters).

> The Chronos-2 code path is covered by a test that uses a tiny random model with the same
> architecture. The real weights could not be downloaded while the starter was built, so check the
> first real run carefully (the forecast chart should follow the clear-sky shape).

## Step 3.6: How to score probabilistic forecasts

Read `src/sahelcast/metrics.py`. All scores use **daylight hours only** (at night every model is
trivially right).

| Score | What it tells you | Good means |
|---|---|---|
| Pinball loss (per quantile) | The proper score for one quantile | Lower |
| Quantile score (mean pinball; ×2 ≈ CRPS) | Overall quality of the whole distribution | Lower |
| MAE of the median | Accuracy of the central forecast | Lower |
| Coverage of the 50/80/90 % ranges | Are the ranges honest? | Close to 50/80/90 % |
| Width of the ranges | How useful (sharp) the ranges are | Narrower, at correct coverage |
| Skill vs persistence | 1 − score ÷ persistence's score | Higher (0 = no better than persistence) |
| Diebold–Mariano test | Is a difference in daily scores statistically real? | p < 0.05 |

Results are also split by **type of day** (clear, mixed, overcast or dusty), because a model can
look good on average and fail exactly on the difficult days.

## Step 3.7: Calibration: make the ranges honest

Read `src/sahelcast/conformal.py` slowly. This is the heart of the project.

**The problem.** Models are rarely honest about their uncertainty. In the demo run made while the
starter was built (synthetic data), LightGBM's national median had only 36 % of observations below
it (it should be 50 %): it was biased low. Its 80 % range covered only 71 % of the hours: it was
overconfident. After calibration the figures were 51 % and 84 %. (The demo's window moves with
today's date, so your demo numbers will differ a little.)

**The fix, in plain words.** For each quantile level τ (0.05, 0.1, …, 0.95), look at the last 60
days: shift the quantile until exactly a fraction τ of recent observations fall below it. For
example, if only 5 % of recent observations fell below the 10 % quantile, lower it until 10 % do.

Three refinements, each added after a real failure in testing:

1. **Each quantile moves separately** (asymmetric conformalized quantile regression). The first
   version only widened ranges symmetrically. That improved coverage but could not fix the biased
   median.
2. **Errors are divided by clear-sky power.** An error of 0.05 is huge at 7 a.m. and small at
   noon; mixing them without scaling makes morning ranges too wide and noon ranges too narrow.
3. **Days are grouped by what the weather forecast expected** ("Mondrian" calibration): forecast
   clear, mixed or cloudy, known at issue time. Without this, the ranges were right on average but
   too wide on clear days and too narrow on cloudy days. That is right for a report and wrong for
   decisions.

Finally, **adaptive conformal inference** nudges the level after every day
(`tau_eff += gamma × (tau − observed share below)`), so calibration recovers quickly after a dust
episode. And the timing is honest: a forecast's errors are only used once its target day has been
observed (`lag_days: 3`, see Step 3.1). The test `test_calibration_never_uses_outcomes_that_are_not_known_yet`
checks this.

## Step 3.8: Run the backtest and read the results

```bash
uv run sahelcast backtest            # the models in settings.yaml, for backtest.start to backtest.end
```

This forecasts every day of 2025 (365 issue days × 19 nodes: 10 plants, 8 wilayas, 1 country),
calibrates, scores, and writes `gold.forecasts`, `gold.metrics`, `gold.metrics_by_regime` and
`gold.significance`. The first 63 days (60 + 3) only warm up the calibration and are not scored.
LightGBM takes a few minutes; Chronos-2 takes longer (use the estimate from Step 3.5).

Read the table it prints, then ask these questions in a notebook:

1. Does every model beat persistence? Is the difference significant (Diebold–Mariano)?
2. After calibration, is coverage within about ±3 points of 80 % and 90 %, overall **and** on
   clear, mixed and dusty days? (In that demo run, calibrated 80 % ranges covered 85 % of
   clear-day hours but only 66 % of the 62 overcast or dusty hours: rare, difficult days are where
   calibration struggles.)
3. Did calibration make the quantile score better or worse? (It can widen ranges and lose sharpness.)
4. Where are the largest errors? Plot the worst ten days. Are they dust events?

**Check:** the tests pass, the metrics table has all models, and you can explain every number in it.

### Phase 3 checkpoint: the MVP
- [ ] A full-year backtest with four models, calibrated, scored by level and by type of day
- [ ] The leakage tests pass
- [ ] Your README "Results" section has your real numbers (see Step 6.5)
- [ ] Commit: "Phase 3: forecasting MVP"

**This is already a strong portfolio project.** Everything after this makes it exceptional.

---

# Phase 4: The hierarchy and a real-data benchmark (weeks 8–9)

## Step 4.1: Make plant, wilaya and national forecasts add up

Each node is forecast on its own, so the national median is usually not the sum of the plant
medians. Sonelgaz would notice. **Reconciliation** adjusts all levels together so they agree:

| Method | Idea |
|---|---|
| Bottom-up | National = sum of plants (ignores the national forecast) |
| MinT (structural) | Combines all levels, trusting each by how many plants it contains |
| MinT (shrinkage) | Combines all levels, trusting each by its past error covariance |

```bash
uv run sahelcast reconcile       # uses the best calibrated model; or --model lightgbm+cqr
```

It reconciles the **medians in MW** with the `hierarchicalforecast` library. The first 60 days only
teach MinT the error covariance; the scores (mean absolute error per level) use the days after.
Quantiles do not add up, so the ranges at each level come from that level's own calibrated
forecast.

**Your turn:** is reconciliation worth it for this hierarchy? With only two wilayas that have more
than one plant, the gains may be small. Say so if that is what you find: an honest null result is
better than a cherry-picked one.

## Step 4.2 (Your turn): Benchmark the method on metered data

The Algerian "observations" are simulated, so a reviewer will ask: *does your method work on real
metered power?* Belgium's grid operator, Elia, publishes national solar output and its own
forecast ranges (10 % and 90 %) every 15 minutes, for free.

The starter has what you need:

- `src/sahelcast/elia.py`: downloads Elia's dataset `ods032`, averages to hours, scores Elia's own
  forecasts, and replaces the simulated national observation with Elia's metered one.
- `config/plants_belgium.csv`: six representative points across Belgium (country code `BE`).
- `scripts/benchmark_elia.py`: builds a separate lake in `data/belgium`, runs the same backtest,
  and prints SahelCast's scores next to Elia's.

```bash
# 1. Check the field names first (open in a browser):
#    https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods032
# 2. Run the benchmark (downloads about two years of data):
uv run python scripts/benchmark_elia.py
```

Two things to do before you trust the result:

1. **Weights.** The six points have equal weight. Elia's dataset has one row per region with its
   monitored capacity; use those to weight the points.
2. **Fairness.** Elia's day-ahead forecast uses fresher weather data than SahelCast's lead-2
   covariates, and SahelCast only uses observations older than 36 hours. If SahelCast comes
   close, that is a strong result; if not, explain why.

> The Elia code was tested against a fake file with the documented field names; it has not been run
> against the real API. If a field name differs, `to_hourly()` stops with the list of real names.

### Phase 4 checkpoint
- [ ] Reconciliation scores per level, with a sentence on whether it helps
- [ ] The Elia benchmark table in your report
- [ ] Commit: "Phase 4: hierarchy and external benchmark"

---

# Phase 5: Decisions: what are the forecasts worth? (weeks 9–11)

## Step 5.1: The decision, in plain words

Each plant promises the grid an hourly delivery for tomorrow (the **commitment**). Every promised
MWh earns the price (1 unit). Every promised MWh that is not delivered costs a **penalty** (3 units
by default), because the operator has to start gas units at short notice. Each plant has a
battery (1 MWh of storage and 0.5 MW of power per MW of plant) that charges from surplus sunshine
and covers shortfalls.

Without a battery, the best promise has a simple answer, the **newsvendor rule**: promise the
forecast quantile at price ÷ penalty = 1 ÷ 3, the 33rd percentile, **not the median**. So the
decision depends directly on the quantiles being right. That is why calibration matters in money,
not just in statistics.

## Step 5.2: The battery plan and the scenarios

Read `src/sahelcast/dispatch.py`:

1. **Scenarios.** Draw 30 possible days of tomorrow's output from the forecast quantiles. Hours are
   linked by a Gaussian copula: a cloudy morning makes a cloudy afternoon more likely. The strength
   of that link is estimated from past forecast errors, before the evaluation period.
2. **Plan.** A linear program (CVXPY, HiGHS solver) chooses one commitment for all scenarios, and a
   battery schedule for each scenario, to maximize the expected profit. The battery must end the
   day as full as it started.
3. **Reality.** Tomorrow's actual output arrives, and the battery follows a simple hour-by-hour rule
   that cannot see the future: charge from surplus, discharge to cover deficits. Shortfalls are paid.

Five policies are compared: perfect foresight (the benchmark: it knows tomorrow's actual output),
planning on the raw median, on the calibrated median, on raw scenarios, and on calibrated
scenarios. **Regret** = perfect
foresight profit − policy profit.

## Step 5.3: Run the study and the sensitivity grid

```bash
uv run sahelcast dispatch --grid   # a 60-day study, then a 3 × 3 grid of battery sizes and penalties
```

The grid solves several thousand small linear programs: expect 5–10 minutes.

How to read it honestly: the value of probabilistic forecasts **depends on the decision**. On the
synthetic demo data (regret in units per MW-day; yours will differ):

- **Calibration pays when shortfalls are expensive.** At a penalty of 5, calibrated scenarios beat
  raw scenarios for every battery size (1-hour battery: 0.32 against 0.37). At lower penalties the
  difference was within noise.
- **The calibrated median was the worst plan at high penalties** (0.45 at a penalty of 5): the
  median is the wrong quantile to promise when missing energy costs five times its price.
- **The raw LightGBM median did best in seven of the nine settings, by accident.** It is biased
  low (about 40 % of plant-level observations fall below it), so it behaves like a cautious
  quantile, which is what the newsvendor rule asks for. Do not read this as "uncertainty is
  useless"; check whether it holds on real data.
- **A bigger battery makes forecasts matter less:** with 2 hours of storage, every policy's regret
  was below 0.18.
- The battery and prices are illustrative. Present them as assumptions, and show the sensitivity
  grid.

**Your turn:** add a sixth policy that commits a calibrated lower quantile (the newsvendor rule,
adjusted for the battery) and see where it lands.

## Step 5.4 (advanced): Decision-focused learning

Usually a forecaster is trained to be accurate, then used for decisions. **Decision-focused
learning** trains it on the cost of the decisions it leads to: the decision is a convex
optimization problem, and `cvxpylayers` makes it differentiable, so the realized cost can be
minimized by gradient descent.

```bash
uv run python scripts/dfl_example.py      # about 1.5 minutes
```

This small example (no battery, toy data) shows the core idea in about 75 lines: on held-out
days, the decision-focused model earned about 3 % more than the least-squares model (4.01 against
3.88 per day), because it learned to promise a little less. **Warm-starting** from the
least-squares model matters: from a random start, the forecast got stuck at zero, where the
gradient vanishes. Published results for PV plus battery report 3.6 % lower costs, or about 8 %
with warm-starting (Depoortere, Kazmi and Driesen, 2026).

**Your turn:** replace the toy data with your backtest's features, and the "no battery" cost with
the real-time battery rule from `dispatch.py` written in PyTorch (it is made of `min` and `max`
operations, which PyTorch can differentiate).

### Phase 5 checkpoint
- [ ] Dispatch summary and sensitivity grid, with your interpretation in two or three sentences
- [ ] (Advanced) a decision-focused result on your own data
- [ ] Commit: "Phase 5: decisions"

---

# Phase 6: Run it every day and show it (weeks 11–12)

## Step 6.1: Tomorrow's forecast

```bash
make daily
```

This runs, in order: `ingest` for the last 10 days, `ingest-live` (the newest forecast run),
`transform`, `physics`, `forecast`, `publish` and `report`. Satellite values (about two days
late) and reanalysis temperatures (a few days late) fill in over the following daily runs; until
they arrive, the observed power uses the weather forecast's temperature. The `forecast` step
reuses the backtest code for the last 70 days plus today, calibrates with the recent days whose
outcomes are known, and keeps today's forecast for tomorrow.

One honest caveat for your write-up: tomorrow's covariates come from the newest forecast run, which
is fresher and usually more accurate than the lead-2 forecasts the models learned from. The live
ranges are therefore slightly cautious.

## Step 6.2: Schedule it with Dagster

```bash
uv run dagster dev          # then open http://127.0.0.1:3000
```

`src/sahelcast/orchestration/definitions.py` defines the same steps as **assets** (so you can see
the lineage graph and each run's metadata) and two schedules:

- `daily_forecast` at 12:15 UTC (13:15 in Algeria)
- `weekly_evaluation` on Sundays at 03:00 UTC (backtest and decisions)

Turn the schedules on in the **Automation** page. Everything runs in one process, one step at a
time, because the lake has one writer.

Your Mac must be awake with `dagster dev` running for schedules to fire. Use
`caffeinate -i uv run dagster dev` while it works. For the portfolio, a week of daily runs,
screenshots of the asset graph and the run history is enough evidence.

## Step 6.3: The API

```bash
uv run sahelcast serve       # then open http://127.0.0.1:8000/docs
```

Endpoints: `/health`, `/plants`, `/forecast/latest?node_id=DZ` (tomorrow's forecast),
`/forecast?issue_date=2025-12-01` (a backtest day; the last 60 are published),
`/metrics?level=country`, `/dispatch/summary`. The API reads small Parquet "serving" tables written by
`sahelcast publish`, never the lake itself, so it keeps working while the pipeline writes. In a
URL, write the `+` of a model name as `%2B` (`model=lightgbm%2Bcqr`); the `/docs` page does this
for you.

## Step 6.4: The website, free on GitHub Pages

```bash
make report                                   # writes site/index.html
mkdir -p docs && cp site/index.html docs/index.html
git add docs && git commit -m "Update website" && git push
```

On GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch:
`main`, folder `/docs` → Save**. After a minute the site is at
`https://<your-username>.github.io/sahelcast/`.

The page is plain HTML with Plotly charts, works on phones, follows the visitor's light or dark
mode, and has a table under every chart. It shows a "Synthetic data" banner whenever it was built
from the demo, so **never publish a demo build as your results**.

## Step 6.5: Write it up

**README results.** Replace the placeholders at the top of `README.md` with your measured numbers
and the period they cover.

**A 4–6 page report** (PDF or a blog post), in this order:

1. The problem: Algeria's solar build-out, and why forecasts need honest uncertainty.
2. Data and the honesty rules: sources, the lead-2 rule, the cutoff, virtual telemetry and its
   validation (PVDAQ, BSRN).
3. Models: persistence, physical, hybrid LightGBM, Chronos-2.
4. Calibration: what failed and why each refinement was needed.
5. Results: scores, coverage by type of day, significance, reconciliation, the Elia benchmark.
6. Decisions: the newsvendor rule, the battery study, the sensitivity grid.
7. Limitations and next steps.

**CV bullets.** Fill the brackets only with numbers you measured:

- Built a DuckLake lakehouse (bronze/silver/gold, time travel) ingesting as-issued NWP forecasts
  (Open-Meteo Previous Runs) and SARAH-3 satellite irradiance for [N] Algerian PV plants, with dbt
  data tests and leak-free backtests enforced by automated leakage tests. (Say "streaming" only if
  you built Step 1.7.)
- Forecast day-ahead PV output with a hybrid physics + LightGBM model and zero-shot Chronos-2,
  improving the quantile score by [X]% over persistence at national level ([period], p < [p]).
- Designed per-quantile, clear-sky-normalized, regime-conditional conformal calibration that
  brought 80% range coverage from [A]% to [B]% on clear, mixed and dusty days alike.
- Quantified the value of calibrated forecasts for PV + battery scheduling with a stochastic LP
  (CVXPY): regret of [R] vs [S] per MW-day at a [penalty]× shortfall penalty.
- Benchmarked the method on metered Belgian grid data against the operator's own forecasts
  (Elia): [result].

**The two-minute interview story.** The question (honest solar forecasts for Algeria), one design
decision (the lead-2 rule and the leakage tests), one hard problem (calibration that was right on
average but wrong on cloudy days, fixed with Mondrian calibration), the result (your numbers),
and what you would do next.

### Final checklist
- [ ] Every phase checkpoint above is ticked
- [ ] CI is green; `make test` passes locally
- [ ] README results use real, measured numbers; the website is live
- [ ] Data credits are in the README and on the website (Open-Meteo CC BY 4.0, EUMETSAT SARAH-3,
      plant sources)
- [ ] You can explain every file in `src/sahelcast/` without reading it

---

# Appendix A: Command cheat sheet

| Command | What it does |
|---|---|
| `make setup` | Install everything and the git hooks |
| `make demo` | The whole pipeline offline on synthetic data (`data/demo`) |
| `make test` / `make test-fast` | All 27 tests / everything except the end-to-end test |
| `uv run sahelcast ingest [--start --end]` | Download history into bronze |
| `uv run sahelcast ingest-live` | Download the newest forecast run |
| `uv run sahelcast transform [dbt args]` | dbt build (or any dbt command, e.g. `docs generate`) |
| `uv run sahelcast physics` | pvlib: `gold.pv_hourly` |
| `uv run sahelcast backtest [--models --start --end]` | Backtest, calibration, scores |
| `uv run sahelcast reconcile [--model]` | Hierarchical reconciliation |
| `uv run sahelcast dispatch [--days --grid]` | Battery decisions and regret |
| `uv run sahelcast forecast` | Tomorrow's forecast |
| `uv run sahelcast publish` / `report` | Serving tables / website |
| `uv run sahelcast serve` | REST API on port 8000 |
| `uv run sahelcast pipeline --skip-ingest` | Every step after ingestion |
| `make daily` | What the daily schedule does (last 10 days, live forecast, website) |
| `uv run python scripts/try_chronos.py 2025-06-01` | Chronos-2 on one week, timed |
| `uv run python scripts/benchmark_elia.py` | The Elia benchmark (Step 4.2) |
| `uv run python scripts/dfl_example.py` | Decision-focused learning example (Step 5.4) |
| `uv run sahelcast snapshots` | The lake's version history |
| `uv run dagster dev` | Scheduler UI on port 3000 |

# Appendix B: Troubleshooting

| Problem | Fix |
|---|---|
| LightGBM: `Library not loaded: libomp.dylib` | `brew install libomp` |
| `Could not set lock on file ... metadata.ducklake` | Another process has the lake open (a Python shell, Dagster, a notebook). Close it; run steps one at a time |
| Open-Meteo `429 Too Many Requests` | Lower `requests_per_minute`, wait an hour (or until tomorrow if you hit the daily limit), run again: the cache resumes |
| Open-Meteo `400` (the error shows the reason) | Wrong model name, variable or dates. Try the URL in a browser (Step 1.2) |
| The newest satellite values are empty | Normal: SARAH-3 is published about two days late. The next daily run fills them in |
| `a backtest needs more than 63 issue days` | Choose a longer period. To try a model quickly, use `scripts/try_chronos.py` |
| `calibration.lag_days=... would use outcomes that are not observed yet` | You changed `obs_cutoff_hours`: raise `lag_days` as the message says |
| dbt: `Env var required but not provided` | Run dbt through `uv run sahelcast transform`, which sets the paths |
| Chronos-2 is slow | `device: mps`, or `chronos_model_id: autogluon/chronos-2-small`, or fewer days |
| MPS error in PyTorch | `export PYTORCH_ENABLE_MPS_FALLBACK=1`, or `device: cpu` |
| Dagster schedule did not run | The Mac was asleep or `dagster dev` was not running |
| The website map is empty | It needs internet (the map outlines load from Plotly's CDN) |

# Appendix C: What was tested, and what you must check

The starter was built and tested in a Linux sandbox that could reach Python's package index but
**not** Open-Meteo, Hugging Face, Elia or AWS. So:

| Part | Status |
|---|---|
| All modules, dbt models and tests, Dagster definitions, API, website (light, dark, phone) | Tested: 27 tests pass; the offline demo runs end to end in about 3 minutes |
| Open-Meteo requests | Written from the official documentation; model names, derived variables and date limits checked against Open-Meteo's open-source server code (September 2026); tested against a synthetic server that returns the same JSON shape. **Check the first real run** (Step 1.2) |
| Chronos-2 | Code path (backtest and `scripts/try_chronos.py`) tested with a tiny random model of the same architecture; real weights not run. **Check the first real forecast** |
| Elia benchmark | Parsing and target replacement tested on a fake file; the live export not run |
| PVDAQ, BSRN, streaming (Redpanda) | Instructions only, written from documentation. **Check each command** |
| Decision-focused example | Runs as shipped (toy data) |

Every number quoted from the starter's development in this guide comes from **synthetic data**. It
explains design choices; it is not a result.

# Appendix D: The $0 checklist

| Item | Cost | Why |
|---|---|---|
| Open-Meteo APIs | $0 | Free for non-commercial use, no key (credit CC BY 4.0) |
| Elia, PVDAQ, BSRN data | $0 | Open data (BSRN: free account, no redistribution) |
| Python packages, DuckDB, DuckLake, dbt Core, Dagster, Chronos-2 | $0 | Open source, run locally |
| Docker for the optional streaming step | $0 | Colima is free and open source (Docker Desktop is also free for personal use) |
| GitHub repository, Actions, Pages | $0 | Free for public repositories on standard runners |
| Compute | $0 | Your Mac; no GPU needed |

Never add a credit card to a cloud "free tier" for this project; nothing here needs one.

# Appendix E: Sources

- Open-Meteo documentation: [Forecast API](https://open-meteo.com/en/docs),
  [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api),
  [Satellite Radiation API](https://open-meteo.com/en/docs/satellite-radiation-api),
  [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api),
  [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api),
  [terms](https://open-meteo.com/en/terms), [server source code](https://github.com/open-meteo/open-meteo)
- SARAH-3 satellite irradiance (EUMETSAT CM SAF): [ESSD paper](https://essd.copernicus.org/articles/16/5243/2024/),
  [EUMETSAT catalogue entry](https://user.eumetsat.int/catalogue/EO:EUM:DAT:0863)
- DuckLake: [ducklake.select](https://ducklake.select/); dbt-duckdb: [GitHub](https://github.com/duckdb/dbt-duckdb)
- pvlib: [pvlib-python.readthedocs.io](https://pvlib-python.readthedocs.io/)
- Chronos-2: [paper](https://arxiv.org/abs/2510.15821), [model](https://huggingface.co/amazon/chronos-2),
  [code](https://github.com/amazon-science/chronos-forecasting)
- Conformalized quantile regression: Romano, Patterson and Candès (2019), [arXiv:1905.03222](https://arxiv.org/abs/1905.03222);
  adaptive conformal inference: Gibbs and Candès (2021), [arXiv:2106.00170](https://arxiv.org/abs/2106.00170)
- Forecast reconciliation: [hierarchicalforecast](https://github.com/Nixtla/hierarchicalforecast);
  MinT: Wickramasuriya, Athanasopoulos and Hyndman (2019)
- Decision-focused learning for PV + battery: Depoortere, Kazmi and Driesen (2026),
  [arXiv:2605.28340](https://arxiv.org/abs/2605.28340); [cvxpylayers](https://github.com/cvxgrp/cvxpylayers)
- Elia open data: [ods032](https://opendata.elia.be/explore/dataset/ods032/);
  NREL PVDAQ: [OEDI](https://data.openei.org/submissions/4568);
  BSRN: [PANGAEA collection](https://doi.pangaea.de/10.1594/PANGAEA.965124)
- Algerian plants: [Global Energy Monitor wiki](https://www.gem.wiki/Ain_El_Ibel_Solar);
  [Enerdata on Tendla and El Ghrous](https://www.enerdata.net/publications/daily-energy-news/algeria-commissions-two-solar-pv-plants-totalling-400-mw.html)
