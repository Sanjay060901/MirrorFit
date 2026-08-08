"""MirrorFit backend — Stage 4 target. Endpoints per spec section 4.6.

You know this stack. The skeleton exists so the repo shape is right from day 1;
build it out when OPERATION_PLAN.md Stage 4 begins. Until then: ignore.
"""
from fastapi import FastAPI

app = FastAPI(title="MirrorFit API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Stage 4 TODOs (see docs/OPERATION_PLAN.md 4.2–4.4):
# GET  /v1/garments/{uid}
# GET  /v1/garments/{uid}/sizes
# POST /v1/fit/recommendation
# GET  /v1/brands/{brand_id}/size-charts
# POST /v1/events
