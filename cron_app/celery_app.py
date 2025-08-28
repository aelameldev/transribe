# celery_app.py
from celery import Celery
from celery.schedules import crontab

celery_app = Celery('celery_app', broker='redis://localhost:6379/0')
celery_app.conf.update(
    result_backend='redis://localhost:6379/0',
    beat_schedule={
        'process-pending-transcriptions': {
            'task': 'tasks.process_pending_transcriptions',
            'schedule': 60.0,  # every 10 seconds
            'args': ()
            # 'schedule': crontab(sec='*/5'),  # every minute
        }
    },
    include=['tasks']
)