from fastapi import FastAPI

from hannah_montana_ai.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hannah-Montana-AI",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(
        router,
        prefix="/api/v1",
    )
    return app


app = create_app()
