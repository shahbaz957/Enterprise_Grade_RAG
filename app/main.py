from fastapi import FastAPI

app = FastAPI(title="Enterprise RAG Backend", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/metrics")
async def metrics() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/graph")
async def graph() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
async def query(payload: dict) -> dict:
    return {"answer": "", "question": payload.get("question", "")}
