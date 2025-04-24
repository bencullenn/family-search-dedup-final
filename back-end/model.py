from sqlalchemy import (
    Boolean,
    Column,
    BigInteger,
    Text,
    TIMESTAMP,
    ForeignKey,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import BIT
from pgvector.sqlalchemy import Vector
from database import Base
import enum


class Person(Base):
    __tablename__ = "person"
    id = Column(BigInteger, primary_key=True)
    created_at = Column(TIMESTAMP(timezone=True))
    first_name = Column(Text)
    last_name = Column(Text)
    folder = Column(Text)


class FlagStatus(str, enum.Enum):
    FLAGGED = "FLAGGED"


class Image(Base):
    __tablename__ = "image"
    id = Column(BigInteger, primary_key=True)
    created_at = Column(TIMESTAMP(timezone=True))
    stack = Column(BigInteger)
    path = Column(Text)
    orb_features = Column(BIT(256))
    image_embedding = Column(Vector(384), nullable=True)
    image_embedding_hq = Column(Vector(1536), nullable=True)
    person = Column(BigInteger, ForeignKey("person.id"))
    title = Column(Text, nullable=True)
    flag_status = Column(SQLEnum(FlagStatus, name="flag_status"), nullable=True)
    flagged_by = Column(BigInteger, ForeignKey("user.id"), nullable=True)
    exclude = Column(Boolean, default=False)


class Stack(Base):
    __tablename__ = "stack"

    id = Column(BigInteger, primary_key=True)
    created_at = Column(TIMESTAMP(timezone=True))
    person = Column(BigInteger, ForeignKey("person.id"))
    name = Column(Text)
    meta_data = Column(Text)
    thumbnail = Column(BigInteger, ForeignKey("image.id"))


class User(Base):
    __tablename__ = "user"
    id = Column(BigInteger, primary_key=True)
