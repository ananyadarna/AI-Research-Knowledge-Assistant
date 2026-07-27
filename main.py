import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from src.database.base import engine, Base
from routes.document_routes import router as doc_router
from routes.search_routes import router as search_router
from routes.analysis_routes import router as analysis_router
from routes.analytics_routes import router as analytics_router

# Setup logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize SQL Database tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("SQL database tables initialized successfully.")
except Exception as e:
    logger.exception(f"Failed to initialize SQL database tables: {e}")

# Instantiate FastAPI
app = FastAPI(
    title="AI Research & Knowledge Assistant API",
    description=(
        "Production-grade backend system facilitating Retrieval-Augmented Generation (RAG) "
        "and automated TensorFlow domain classification for PDF documents."
    ),
    version="1.0.0"
)

# Setup CORS middleware parameters
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production environments to specified domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register routers
app.include_router(doc_router)
app.include_router(search_router)
app.include_router(analysis_router)
app.include_router(analytics_router)

@app.get("/", tags=["root"])
def read_root():
    return {
        "message": "Welcome to the AI Research & Knowledge Assistant API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server at http://{settings.host}:{settings.port}")
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
