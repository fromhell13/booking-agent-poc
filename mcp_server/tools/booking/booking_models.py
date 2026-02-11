from __future__ import annotations
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, Boolean

class Base(DeclarativeBase):
    pass

class Reservation(Base):
    __tablename__ = "reservations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120))
    phone: Mapped[str] = mapped_column(String(50))
    date: Mapped[str] = mapped_column(String(10))   
    time: Mapped[str] = mapped_column(String(5))    
    pax: Mapped[int] = mapped_column(Integer)
    notes: Mapped[str] = mapped_column(String(500), default="")
    cancelled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
