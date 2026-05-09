from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# Define a user model
class User(BaseModel):
    id: int
    username: str = Field(min_length=3, max_length=20)
    email: str = Field(pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    age: Optional[int] = Field(gt=0, lt=120)
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)

# Create a user instance
user_data = {
    "id": 1,
    "username": "testuser",
    "email": "test@example.com",
    "age": 25,
    "tags": ["python", "developer"]
}

# Validate data
try:
    user = User(**user_data)
    print("User data validation successful!")
    print(f"Username: {user.username}")
    print(f"Email: {user.email}")
    print(f"Age: {user.age}")
    print(f"Created at: {user.created_at}")
    print(f"Tags: {user.tags}")

    # Convert to dictionary
    print("\nConvert to dictionary:")
    print(user.model_dump())

    # Convert to JSON
    print("\nConvert to JSON:")
    print(user.model_dump_json())

except Exception as e:
    print(f"Data validation failed: {e}")
