"""Authentication Service.

Handles user authentication, password management, and API key generation.
"""

import logging
import secrets
from typing import Optional
from sqlalchemy.orm import Session

import bcrypt

from dbnotebook.core.db.models import User

logger = logging.getLogger(__name__)


class AuthService:
    """Service for user authentication and password management."""

    def __init__(self, db_session: Session):
        """Initialize auth service with database session.

        Args:
            db_session: SQLAlchemy session for database operations
        """
        self.session = db_session

    def login(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password.

        Args:
            username: User's username
            password: Plain text password

        Returns:
            User object if authentication successful, None otherwise
        """
        user = self.session.query(User).filter(User.username == username).first()

        if not user:
            logger.warning(f"Login attempt for non-existent user: {username}")
            return None

        if not user.password_hash:
            logger.warning(f"User {username} has no password set")
            return None

        if self.verify_password(password, user.password_hash):
            logger.info(f"User {username} logged in successfully")
            return user

        logger.warning(f"Invalid password for user: {username}")
        return None

    def change_password(
        self, user_id: str, old_password: str, new_password: str
    ) -> bool:
        """Change user's password.

        Args:
            user_id: User's UUID
            old_password: Current password for verification
            new_password: New password to set

        Returns:
            True if password changed successfully, False otherwise
        """
        user = self.session.query(User).filter(User.user_id == user_id).first()

        if not user:
            logger.warning(f"Password change attempt for non-existent user: {user_id}")
            return False

        if not self.verify_password(old_password, user.password_hash or ""):
            logger.warning(f"Invalid old password for user: {user.username}")
            return False

        user.password_hash = self.hash_password(new_password)
        self.session.commit()
        logger.info(f"Password changed for user: {user.username}")
        return True

    def set_password(self, user_id: str, new_password: str) -> bool:
        """Set user's password (admin operation, no old password required).

        Args:
            user_id: User's UUID
            new_password: New password to set

        Returns:
            True if password set successfully, False otherwise
        """
        user = self.session.query(User).filter(User.user_id == user_id).first()

        if not user:
            logger.warning(f"Set password attempt for non-existent user: {user_id}")
            return False

        user.password_hash = self.hash_password(new_password)
        self.session.commit()
        logger.info(f"Password set for user: {user.username}")
        return True

    def generate_api_key(self, user_id: str) -> Optional[str]:
        """Generate new API key for user.

        Args:
            user_id: User's UUID

        Returns:
            New API key if successful, None otherwise
        """
        user = self.session.query(User).filter(User.user_id == user_id).first()

        if not user:
            logger.warning(f"API key generation for non-existent user: {user_id}")
            return None

        # Generate new API key: dbn_ + 32 random hex chars
        new_api_key = f"dbn_{secrets.token_hex(16)}"
        user.api_key = new_api_key
        self.session.commit()
        logger.info(f"New API key generated for user: {user.username}")
        return new_api_key

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID.

        Args:
            user_id: User's UUID

        Returns:
            User object if found, None otherwise
        """
        return self.session.query(User).filter(User.user_id == user_id).first()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username.

        Args:
            username: User's username

        Returns:
            User object if found, None otherwise
        """
        return self.session.query(User).filter(User.username == username).first()

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        generate_api_key: bool = True
    ) -> Optional[User]:
        """Create new user.

        Args:
            username: Unique username
            email: User's email address
            password: Plain text password
            generate_api_key: Whether to generate API key for new user

        Returns:
            New User object if successful, None if username/email exists
        """
        # Check if username exists
        existing = self.session.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing:
            logger.warning(f"Cannot create user, username or email already exists: {username}")
            return None

        user = User(
            username=username,
            email=email,
            password_hash=self.hash_password(password),
            api_key=f"dbn_{secrets.token_hex(16)}" if generate_api_key else None
        )

        self.session.add(user)
        self.session.commit()
        logger.info(f"Created new user: {username}")
        return user

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Bcrypt hash of password
        """
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash.

        Args:
            password: Plain text password
            password_hash: Bcrypt hash to verify against

        Returns:
            True if password matches, False otherwise
        """
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False
