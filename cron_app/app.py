import asyncio
from db.pool import init_pool
from flask import Flask, jsonify
from tasks import process_pending_transcriptions 

def create_app():
    app = Flask(__name__)
    # await init_pool(min_size=2, max_size=10)

    return app

app = create_app()


@app.post("/process_transcriptions")
def trigger_processing():
    task = process_pending_transcriptions.delay()
    return jsonify({"task_id": task.id, "status": "queued"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5500)