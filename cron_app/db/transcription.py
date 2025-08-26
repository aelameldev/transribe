
import psycopg

import logging
import os

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/cmgmtDB")


SQL_FETCH_PENDING = """
    SELECT id, url
    FROM transcriptions
    WHERE status = 'PROCESSING'
    ORDER BY id
    LIMIT $1
    FOR UPDATE SKIP LOCKED
"""

SQL_MARK_PROCESSING = """
    UPDATE transcriptions
    SET status = 'PROCESSING'
    WHERE id = $1
"""

SQL_MARK_DONE = """
    UPDATE transcriptions
    SET status = 'DONE', transcript = $2
    WHERE id = $1
"""
def run_transcription(url: str) -> str:
    return f"Transcript of {url}"

def process_requests(limit: int = 5):
    conn = psycopg.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(SQL_FETCH_PENDING, (limit,))
                rows = cur.fetchall()

                for row in rows:
                    req_id, url = row["id"], row["url"]
                    logger.info(f"Picked request id={req_id}, file={url}")

                    cur.execute(SQL_MARK_PROCESSING, (req_id,))
                    transcript = run_transcription(url)
                    cur.execute(SQL_MARK_DONE, (transcript, req_id))

                    logger.info(f"✅ Finished request id={req_id}")

    finally:
        conn.close()

