import pandas as pd

from sahelcast.lake import Lake


def test_upsert_is_idempotent_updates_rows_and_keeps_history(settings):
    rows = pd.DataFrame(
        {
            "plant_id": ["a", "b"],
            "valid_time": pd.to_datetime(["2025-01-01 12:00", "2025-01-01 12:00"], utc=True),
            "ghi": [500.0, 600.0],
        }
    )
    with Lake.from_settings(settings) as lake:
        lake.upsert("bronze.t", rows, ["plant_id", "valid_time"])
        lake.upsert("bronze.t", rows, ["plant_id", "valid_time"])  # same load again
        assert len(lake.read("bronze.t")) == 2
        before = int(lake.snapshots()["snapshot_id"].max())

        changed = rows.assign(ghi=[555.0, 600.0])
        lake.upsert("bronze.t", changed, ["plant_id", "valid_time"])
        now = lake.read("bronze.t").set_index("plant_id")["ghi"]
        assert now["a"] == 555.0

        # Time travel: the table as it was before the update.
        old = lake.sql(
            f"SELECT ghi FROM lake.bronze.t AT (VERSION => {before}) WHERE plant_id = 'a'"
        )
        assert old["ghi"].iloc[0] == 500.0
