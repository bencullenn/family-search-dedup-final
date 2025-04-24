from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from enum import Enum


class UserBase(BaseModel):
    name: str


class UserCreate(UserBase):
    pass


class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True


class PersonBase(BaseModel):
    first_name: str
    last_name: str


class PersonCreate(PersonBase):
    pass


class Person(PersonBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
        



class FlagStatus(str, Enum):
    FLAGGED = "FLAGGED"

class ImageBase(BaseModel):
    collection: int
    path: str
    orb_features: List[float]
    flag_status: Optional[FlagStatus] = None
    flagged_by: Optional[int] = None


class ImageCreate(ImageBase):
    pass


class Image(ImageBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True


class StackBase(BaseModel):
    person: int
    name: str
    metadata: str
    thumbnail: int


class StackCreate(StackBase):
    pass


class Stack(StackBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
