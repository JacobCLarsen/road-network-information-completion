from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.api.routes import model_metrics, model_types, model_download, health
from src.db.config import settings
from src.db.engine import engine, check_database
from src.db.exceptions import NotFoundError
from src.api.exceptions import NotFoundException, InvalidUUIDException, ForeignKeyViolationException
from contextlib import asynccontextmanager
import logging
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO if os.getenv("ENV") == "cluster" else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: the tables are made by the migrations in the data repo, so we only check that
    # the database is there and up to date (the API stops with an error if not)
    if settings.db_check_migrations and not settings.database_url.startswith("sqlite"):
        logger.info("Application startup - checking the database")
        await check_database()
        logger.info("Database is ready")
    yield
    # Shutdown: Close connections
    logger.info("Application shutdown - closing database connections")
    await engine.dispose()
    logger.info("Database connections closed")

app = FastAPI(lifespan=lifespan)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path}")

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(f"Response: {response.status_code} | Duration: {duration:.3f}s")

    return response

# Exception handlers
@app.exception_handler(NotFoundException)
async def not_found_handler(request: Request, exc: NotFoundException):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc) if str(exc) else "Resource not found"}
    )

@app.exception_handler(InvalidUUIDException)
async def invalid_uuid_handler(request: Request, exc: InvalidUUIDException):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc) if str(exc) else "Invalid UUID format"}
    )

@app.exception_handler(ForeignKeyViolationException)
async def foreign_key_violation_handler(request: Request, exc: ForeignKeyViolationException):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(NotFoundError)
async def db_not_found_handler(request: Request, exc: NotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid input: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # Log the error for debugging in cluster logs
    logger.error(f"Unexpected error: {type(exc).__name__}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

app.include_router(model_types.router)
app.include_router(model_metrics.router)
app.include_router(model_download.router)
app.include_router(health.router)

# uvicorn src.api.main:app --reload
