import uvicorn


def start_api():
    uvicorn.run(
        "app.api.server:app",
        host="0.0.0.0",
        port=4001,
        log_level="info",
    )