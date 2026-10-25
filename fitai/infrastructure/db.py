from dataclasses import dataclass
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker


@dataclass(frozen=True)
class Database:
    engine: Engine
    session_factory: sessionmaker

    def close(self) -> None:
        self.engine.dispose()


def get_db_path() -> Path:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return data_home / "fitcalc" / "fitai.sqlite3"


def create_database(db_path: str | Path | None = None) -> Database:
    path = Path(db_path) if db_path is not None else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    url = URL.create("sqlite", database=str(path))
    engine = create_engine(url, connect_args={"timeout": 30})

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return Database(
        engine=engine,
        session_factory=sessionmaker(bind=engine, expire_on_commit=False),
    )


def migrate_database(database: Database) -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "migrations"))
    config.set_main_option(
        "sqlalchemy.url",
        database.engine.url.render_as_string(hide_password=False).replace("%", "%%"),
    )
    command.upgrade(config, "head")


def initialize_database(db_path: str | Path | None = None) -> Database:
    database = create_database(db_path)
    try:
        migrate_database(database)
    except Exception:
        database.close()
        raise
    return database
