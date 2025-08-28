import asyncio
from celery.utils.log import get_task_logger
from celery_app import celery_app

from db.pool import init_pool
from db.transcription import process_requests


logger = get_task_logger(__name__)

asyncio.run(init_pool())


@celery_app.task(bind=True,
             autoretry_for=(Exception,),
             retry_kwargs={"max_retries": 5, "countdown": 30},
             retry_backoff=True,
             retry_backoff_max=300)
def process_pending_transcriptions(_) -> int:
    logger.info("Processing pending transcriptions...")
    process_requests()
    return "Task completed"
