import json
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from db.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    status = Column(String, nullable=False, default="created")
    # status values: created | running | complete | error | stopped
    notes = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Summary fields (filled on session complete)
    total_plants = Column(Integer, nullable=True)
    avg_height_cm = Column(Float, nullable=True)
    avg_moisture_pct = Column(Float, nullable=True)
    total_water_sec = Column(Float, nullable=True)
    ripe_count = Column(Integer, nullable=True)
    turning_count = Column(Integer, nullable=True)
    unripe_count = Column(Integer, nullable=True)
    broken_count = Column(Integer, nullable=True)
    harvest_ready_ids = Column(Text, nullable=True)  # JSON list of plant_ids

    plant_scans = relationship(
        "PlantScan", back_populates="session", cascade="all, delete-orphan"
    )

    def harvest_ready_list(self) -> list[int]:
        if not self.harvest_ready_ids:
            return []
        return json.loads(self.harvest_ready_ids)


class PlantScan(Base):
    __tablename__ = "plant_scans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    plant_id = Column(Integer, nullable=False)  # 1..16
    row = Column(Integer, nullable=False)
    col = Column(Integer, nullable=False)
    scanned_at = Column(DateTime(timezone=True), default=utcnow)

    image_url = Column(String, nullable=True)
    total_fruits = Column(Integer, nullable=True)
    ripe_count = Column(Integer, nullable=True, default=0)
    turning_count = Column(Integer, nullable=True, default=0)
    unripe_count = Column(Integer, nullable=True, default=0)
    broken_count = Column(Integer, nullable=True, default=0)
    detections_json = Column(Text, nullable=True)

    height_cm = Column(Float, nullable=True)
    moisture_pct = Column(Float, nullable=True)
    valve_duration_sec = Column(Float, nullable=True)
    watering_reason = Column(String, nullable=True)

    session = relationship("Session", back_populates="plant_scans")

    def detections(self) -> list[dict]:
        if not self.detections_json:
            return []
        return json.loads(self.detections_json)
