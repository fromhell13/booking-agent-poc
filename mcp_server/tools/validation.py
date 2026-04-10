"""
Input validation for all MCP tools. All user-supplied inputs must be validated here.
"""
from __future__ import annotations

import re
from typing import NoReturn

# Date YYYY-MM-DD
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Time HH:MM (24h)
TIME_RE = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")
# Phone: digits, spaces, optional +
PHONE_RE = re.compile(r"^\+?[\d\s\-]{6,20}$")


def _err(msg: str) -> NoReturn:
    raise ValueError(msg)


def validate_date(date: str) -> str:
    if not date or not isinstance(date, str):
        _err("date is required and must be a string")
    date = date.strip()
    if not DATE_RE.match(date):
        _err("date must be YYYY-MM-DD")
    return date


def validate_time(time: str) -> str:
    if not time or not isinstance(time, str):
        _err("time is required and must be a string")
    time = time.strip()
    if not TIME_RE.match(time):
        _err("time must be HH:MM (24-hour)")
    return time


def validate_phone(phone: str) -> str:
    if not phone or not isinstance(phone, str):
        _err("phone is required and must be a string")
    phone = phone.strip()
    if not PHONE_RE.match(phone):
        _err("phone must be 6–20 digits (spaces/dashes allowed)")
    return phone


def validate_name(name: str) -> str:
    if not name or not isinstance(name, str):
        _err("name is required and must be a string")
    name = name.strip()
    if len(name) < 1 or len(name) > 120:
        _err("name must be 1–120 characters")
    return name


def validate_pax(pax: int) -> int:
    if not isinstance(pax, int):
        _err("pax must be an integer")
    if pax < 1 or pax > 50:
        _err("pax must be between 1 and 50")
    return pax


def validate_reservation_id(reservation_id: int) -> int:
    if not isinstance(reservation_id, int):
        _err("reservation_id must be an integer")
    if reservation_id < 1:
        _err("reservation_id must be a positive integer")
    return reservation_id


def validate_query(query: str) -> str:
    if not query or not isinstance(query, str):
        _err("query is required and must be a string")
    query = query.strip()
    if len(query) > 500:
        _err("query must be at most 500 characters")
    return query


def validate_top_k(top_k: int, max_k: int = 20) -> int:
    if not isinstance(top_k, int):
        _err("top_k must be an integer")
    if top_k < 1 or top_k > max_k:
        _err(f"top_k must be between 1 and {max_k}")
    return top_k


def validate_max_tables(max_tables: int) -> int:
    if not isinstance(max_tables, int):
        _err("max_tables must be an integer")
    if max_tables < 1 or max_tables > 100:
        _err("max_tables must be between 1 and 100")
    return max_tables


def validate_notes(notes: str) -> str:
    if notes is None:
        return ""
    if not isinstance(notes, str):
        _err("notes must be a string")
    return notes.strip()[:500]
