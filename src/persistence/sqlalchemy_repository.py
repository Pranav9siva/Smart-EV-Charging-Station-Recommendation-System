from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.persistence.sqlalchemy_models import Base


class SQLAlchemyRepository:
    """Thin SQLAlchemy-backed repository wrapper with SQLite default."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or "sqlite:///outputs/sqlalchemy_app.db"
        self.engine = create_engine(self.database_url)
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return Session(self.engine)

    def save(self, model: Any) -> None:
        with self.session() as session:
            session.add(model)
            session.commit()
