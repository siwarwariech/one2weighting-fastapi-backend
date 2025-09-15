from pydantic import BaseModel, ConfigDict, EmailStr, constr, field_validator
from typing import Optional
from datetime import datetime

# Password complexity requirements
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 50

class SignUpRequest(BaseModel):
    first_name: constr(min_length=1, max_length=50, strip_whitespace=True) 
    last_name: constr(min_length=1, max_length=50, strip_whitespace=True)
    company_name: constr(min_length=1, max_length=100, strip_whitespace=True)
    email: EmailStr
    password: constr(min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH)
    
    @field_validator('password')
    def password_complexity(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*()' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v

class SignInRequest(BaseModel):
    # Note: FastAPI's OAuth2PasswordRequestForm uses 'username' not 'email'
    # This should match what your frontend sends
    email: EmailStr
    password: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123!"
            }
        }
    )

class UserResponse(BaseModel):
    id: int
    first_name: str
    last_name: str
    company_name: str
    email: EmailStr
    created_at: datetime
    is_active: Optional[bool] = True
    
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOi...",
                "token_type": "bearer",
                "expires_in": 3600
            }
        }
    )