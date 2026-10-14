"""The DuckLake lakehouse: one catalog file plus Parquet data files, queried with DuckDB.

Layers:
- bronze: data as received from the APIs (typed, never edited)
- silver: cleaned and quality-checked (built by dbt)
- gold:   joined, model-ready tables and outputs (dbt + Python)

Every change creates a snapshot, so you can query any table as it was earlier:
    SELECT * FROM lake.bronze.satellite AT (VERSION => 3)
That is what makes leak-free backtests auditable.

With a DuckDB file as the catalog, only one process can write at a time. Run the pipeline
steps one after another (the Makefile and Dagster job do). To have several writers, move
the catalog to PostgreSQL: ATTACH 'ducklake:postgres:dbname=sahelcast host=localhost' ...
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import Settings

SCHEMAS = ("bronze", "silver", "gold")


class Lake:
    def __init__(self, metadata_path: Path, data_path: Path, read_only: bool = False) -> None:
        metadata_path = Path(metadata_path)
        data_path = Path(data_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect()
        self.con.execute("SET TimeZone = 'UTC'")
        self.con.execute("INSTALL ducklake")
        self.con.execute("LOAD ducklake")
        options = f"DATA_PATH '{data_path.as_posix()}/'"
        if read_only:
            options += ", READ_ONLY"
        self.con.execute(f"ATTACH 'ducklake:{metadata_path.as_posix()}' AS lake ({options})")
        if not read_only:
            for schema in SCHEMAS:
                self.con.execute(f"CREATE SCHEMA IF NOT EXISTS lake.{schema}")

    @classmethod
    def from_settings(cls, settings: Settings, read_only: bool = False) -> Lake:
        return cls(settings.paths.lake_metadata, settings.paths.lake_data, read_only)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Lake:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- reading
    def sql(self, query: str, params: list | None = None) -> pd.DataFrame:
        return self.con.execute(query, params or []).df()

    def table_exists(self, table: str) -> bool:
        schema, name = table.split(".")
        found = self.con.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE database_name = 'lake' "
            "AND schema_name = ? AND table_name = ?",
            [schema, name],
        ).fetchone()
        return bool(found and found[0])

    def read(self, table: str, where: str = "") -> pd.DataFrame:
        return self.sql(f"SELECT * FROM lake.{table} {where}")

    def snapshots(self) -> pd.DataFrame:
        return self.sql("SELECT * FROM lake.snapshots()")

    # ---------------------------------------------------------------- writing
    def upsert(self, table: str, frame: pd.DataFrame, keys: list[str]) -> int:
        """Insert new rows and update existing ones (matched on `keys`).

        Running the same load twice leaves the table unchanged: the same idea as your
        PostgreSQL upserts (INSERT ... ON CONFLICT), written as a MERGE.
        """
        if frame.empty:
            return 0
        frame = frame.drop_duplicates(subset=keys, keep="last")
        self.con.register("incoming", frame)
        try:
            if not self.table_exists(table):
                self.con.execute(f"CREATE TABLE lake.{table} AS SELECT * FROM incoming LIMIT 0")
            on = " AND ".join(f"t.{k} IS NOT DISTINCT FROM s.{k}" for k in keys)
            updates = ", ".join(f"{c} = s.{c}" for c in frame.columns if c not in keys)
            matched = f"WHEN MATCHED THEN UPDATE SET {updates}" if updates else ""
            self.con.execute(
                f"MERGE INTO lake.{table} AS t USING incoming AS s ON {on} "
                f"{matched} WHEN NOT MATCHED THEN INSERT BY NAME"
            )
        finally:
            self.con.unregister("incoming")
        return len(frame)

    def replace(self, table: str, frame: pd.DataFrame) -> int:
        """Overwrite a whole table (used for derived gold tables)."""
        self.con.register("incoming", frame)
        try:
            self.con.execute(f"CREATE OR REPLACE TABLE lake.{table} AS SELECT * FROM incoming")
        finally:
            self.con.unregister("incoming")
        return len(frame)
