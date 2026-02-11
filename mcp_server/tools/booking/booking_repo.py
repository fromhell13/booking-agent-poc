from __future__ import annotations
import os
from sqlalchemy import create_engine, select, and_
from sqlalchemy.orm import sessionmaker

from .booking_models import Base, Reservation

DATABASE_URL = os.getenv("DATABASE_URL", "")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db() -> None:
    Base.metadata.create_all(bind=engine)

def create_reservation(name: str, phone: str, date: str, time: str, pax: int, notes: str = "") -> dict:
    with SessionLocal() as db:
        r = Reservation(name=name, phone=phone, date=date, time=time, pax=pax, notes=notes)
        db.add(r)
        db.commit()
        db.refresh(r)
        return {"id": r.id, "name": r.name, "phone": r.phone, "date": r.date, "time": r.time, "pax": r.pax, "notes": r.notes}

def list_reservations(date: str | None = None) -> dict:
    with SessionLocal() as db:
        stmt = select(Reservation).where(Reservation.cancelled == False)  # noqa: E712
        if date:
            stmt = stmt.where(Reservation.date == date)
        rows = db.execute(stmt).scalars().all()
        return {"items": [{"id": r.id, "name": r.name, "phone": r.phone, "date": r.date, "time": r.time, "pax": r.pax, "notes": r.notes} for r in rows]}

def cancel_reservation(reservation_id: int) -> dict:
    with SessionLocal() as db:
        r = db.get(Reservation, reservation_id)
        if not r:
            return {"ok": False, "error": "not_found"}
        r.cancelled = True
        db.commit()
        return {"ok": True, "id": reservation_id}

def check_availability(date: str, time: str, max_tables: int = 10) -> dict:
    # Simple PoC rule: each reservation consumes 1 table
    with SessionLocal() as db:
        stmt = select(Reservation).where(
            and_(
                Reservation.cancelled == False, 
                Reservation.date == date,
                Reservation.time == time,
            )
        )
        count = len(db.execute(stmt).scalars().all())
        remaining = max_tables - count
        return {"date": date, "time": time, "reserved_tables": count, "remaining_tables": max(0, remaining), "available": remaining > 0}
