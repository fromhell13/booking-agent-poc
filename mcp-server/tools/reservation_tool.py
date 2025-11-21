# tools/reservation_tool.py

import os
import psycopg2
import re

# Assuming a reservation table schema like:
# CREATE TABLE reservations (
#     id SERIAL PRIMARY KEY,
#     name TEXT,
#     date DATE,
#     time TIME,
#     people INTEGER
# );

def parse_query(query: str):
    """Naive regex parser. Replace with LLM later if needed."""
    import datetime
    match = re.search(r'(\d+)\s*people.*?on\s*(\d{4}-\d{2}-\d{2})\s*at\s*(\d{1,2}:\d{2})', query.lower())
    if not match:
        return None

    people = int(match.group(1))
    date = match.group(2)
    time = match.group(3)
    return people, date, time

def check_table_availability(query: str) -> str:
    parsed = parse_query(query)
    if not parsed:
        return "Sorry, I couldn't understand the reservation details. Please specify people, date, and time."

    people, date, time = parsed

    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=os.getenv("POSTGRES_PORT", 5432),
            database=os.getenv("POSTGRES_DB", "reservations"),
            user=os.getenv("POSTGRES_USER", "user"),
            password=os.getenv("POSTGRES_PASSWORD", "pass")
        )

        cur = conn.cursor()

        # Example logic: allow max 10 tables per slot
        cur.execute("""
            SELECT COUNT(*) FROM reservations
            WHERE date = %s AND time = %s;
        """, (date, time))

        current_count = cur.fetchone()[0]
        max_tables = 10

        if current_count >= max_tables:
            return f"Sorry, we're fully booked on {date} at {time}."

        # (Optional) auto-insert dummy reservation for demo purposes
        cur.execute("""
            INSERT INTO reservations (name, date, time, people)
            VALUES (%s, %s, %s, %s)
        """, ("AI Chatbot", date, time, people))

        conn.commit()
        return f"✅ Table reserved for {people} on {date} at {time}."

    except Exception as e:
        return f"Database error: {e}"

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
