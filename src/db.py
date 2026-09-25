"""Shared DuckDB connection and schema helpers."""

import duckdb

from config import DB_PATH


def _initialize_schema(connection: duckdb.DuckDBPyConnection) -> None:
	"""Create the pipeline tables if they do not already exist."""
	connection.execute(
		"""
		CREATE TABLE IF NOT EXISTS news (
			id VARCHAR PRIMARY KEY,
			ticker VARCHAR NOT NULL,
			title VARCHAR NOT NULL,
			source VARCHAR NOT NULL,
			url VARCHAR NOT NULL,
			published_at TIMESTAMPTZ NOT NULL
		)
		"""
	)
	connection.execute(
		"""
		CREATE TABLE IF NOT EXISTS sentiment (
			headline_id VARCHAR PRIMARY KEY,
			sentiment VARCHAR NOT NULL CHECK (
				sentiment IN ('positive', 'neutral', 'negative')
			),
			score DOUBLE NOT NULL CHECK (score >= -1 AND score <= 1),
			topic VARCHAR NOT NULL,
			scored_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
		)
		"""
	)


def get_connection() -> duckdb.DuckDBPyConnection:
	"""Return a connection to the project's file-backed DuckDB database."""
	DB_PATH.parent.mkdir(parents=True, exist_ok=True)
	connection = duckdb.connect(str(DB_PATH))
	_initialize_schema(connection)
	return connection
    

