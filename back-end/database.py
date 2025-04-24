from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
import os

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DB_CONN_STRING_POOL")

# Set up an engine that doesn't pool connections client side
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    poolclass=NullPool,  # Disable SQLAlchemy's internal pooling
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
