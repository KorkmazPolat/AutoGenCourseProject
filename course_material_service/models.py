from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, JSON, Boolean, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from course_material_service.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    enrollments = relationship("Enrollment", back_populates="user", cascade="all, delete-orphan")
    created_courses = relationship("Course", back_populates="creator", cascade="all, delete-orphan")
    lesson_completions = relationship("LessonCompletion", back_populates="user", cascade="all, delete-orphan")
    quiz_attempts = relationship("QuizAttempt", back_populates="user", cascade="all, delete-orphan")


class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text, nullable=True)
    learning_outcomes = Column(JSON)  # List of strings
    thumbnail_url = Column(String, nullable=True)
    is_published = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # Nullable for migration safety, but should be populated
    course_type = Column(String, default="course") # 'course', 'quiz', 'reading', 'slides'

    creator = relationship("User", back_populates="created_courses")
    modules = relationship("CourseModule", back_populates="course", cascade="all, delete-orphan")
    enrollments = relationship("Enrollment", back_populates="course", cascade="all, delete-orphan")


class Enrollment(Base):
    __tablename__ = "enrollments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    course_id = Column(Integer, ForeignKey("courses.id"))
    enrolled_at = Column(DateTime(timezone=True), server_default=func.now())
    progress_percent = Column(Float, default=0.0)

    user = relationship("User", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")


class CourseModule(Base):
    __tablename__ = "modules"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    title = Column(String)
    summary = Column(Text, nullable=True)
    order_index = Column(Integer)

    course = relationship("Course", back_populates="modules")
    lessons = relationship("Lesson", back_populates="module", cascade="all, delete-orphan")


class Lesson(Base):
    __tablename__ = "lessons"

    id = Column(Integer, primary_key=True, index=True)
    module_id = Column(Integer, ForeignKey("modules.id"))
    title = Column(String)
    content = Column(Text)  # The markdown content
    order_index = Column(Integer)
    duration_minutes = Column(Integer, default=10)

    module = relationship("CourseModule", back_populates="lessons")
    assets = relationship("LessonAsset", back_populates="lesson", cascade="all, delete-orphan")
    completions = relationship("LessonCompletion", back_populates="lesson", cascade="all, delete-orphan")


class LessonAsset(Base):
    __tablename__ = "lesson_assets"

    id = Column(Integer, primary_key=True, index=True)
    lesson_id = Column(Integer, ForeignKey("lessons.id"))
    asset_type = Column(String)  # 'script', 'video', 'quiz'
    content = Column(JSON, nullable=True)  # For script/quiz data
    file_path = Column(String, nullable=True)  # For video path
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    lesson = relationship("Lesson", back_populates="assets")
    quiz_attempts = relationship("QuizAttempt", back_populates="asset", cascade="all, delete-orphan")


class LessonCompletion(Base):
    __tablename__ = "lesson_completions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    lesson_id = Column(Integer, ForeignKey("lessons.id"))
    completed_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="lesson_completions")
    lesson = relationship("Lesson", back_populates="completions")


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    asset_id = Column(Integer, ForeignKey("lesson_assets.id")) # Must be a quiz asset
    score = Column(Float)
    max_score = Column(Float)
    attempted_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="quiz_attempts")
    asset = relationship("LessonAsset", back_populates="quiz_attempts")
