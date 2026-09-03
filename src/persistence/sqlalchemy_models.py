from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Vehicle(Base):
    __tablename__ = "vehicles"
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(String(255), nullable=False, unique=True)
    battery_pct = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChargingSession(Base):
    __tablename__ = "charging_sessions"
    id = Column(Integer, primary_key=True)
    station_id = Column(String(255), nullable=False)
    vehicle_id = Column(String(255), nullable=False)
    status = Column(String(255), default="charging")
    created_at = Column(DateTime, default=datetime.utcnow)


class Station(Base):
    __tablename__ = "stations"
    id = Column(Integer, primary_key=True)
    station_id = Column(String(255), nullable=False, unique=True)
    available_ports = Column(Integer, default=0)
    price_per_kwh = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(String(255), nullable=False)
    station_id = Column(String(255), nullable=False)
    confidence = Column(Float, default=0.0)
    reward = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Episode(Base):
    __tablename__ = "episodes"
    id = Column(Integer, primary_key=True)
    episode_id = Column(String(255), nullable=False, unique=True)
    reward = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Metrics(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    value = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), nullable=False)
    version = Column(String(255), nullable=False)
    metadata_payload = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)


class Explanation(Base):
    __tablename__ = "explanations"
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(String(255), nullable=False)
    station_id = Column(String(255), nullable=False)
    payload = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)


class SimulationHistory(Base):
    __tablename__ = "simulation_history"
    id = Column(Integer, primary_key=True)
    scenario_name = Column(String(255), nullable=False)
    payload = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)
