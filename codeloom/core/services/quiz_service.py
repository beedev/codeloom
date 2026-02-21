"""Quiz service implementation for DBProject.

This module implements the quiz service layer, handling quiz creation,
question generation via LLM, adaptive difficulty, and result tracking.

Supports:
- Standard multiple-choice questions from project content
- Extended questions that go beyond project content (inferred topics)
- Code-based questions (output prediction, fill-in-blank, bug identification)
"""

import json
import os
import uuid
import re
import random
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List

from sqlalchemy import select, func, desc
from sqlalchemy.orm.attributes import flag_modified

from .base import BaseService
from ..db.models import Quiz, QuizAttempt, Project, CodeFile


# Difficulty level constants
DIFFICULTY_EASY = 1
DIFFICULTY_MEDIUM = 2
DIFFICULTY_HARD = 3

DIFFICULTY_LABELS = {
    DIFFICULTY_EASY: 'easy',
    DIFFICULTY_MEDIUM: 'medium',
    DIFFICULTY_HARD: 'hard'
}


class QuestionType(str, Enum):
    """Types of quiz questions supported."""
    MULTIPLE_CHOICE = "multiple_choice"      # Standard 4-option question
    CODE_OUTPUT = "code_output"              # What's the output of this code?
    CODE_FILL_BLANK = "code_fill_blank"      # Fill in the missing code
    CODE_BUG_FIX = "code_bug_fix"            # Identify the bug in this code

# Question generation prompt template
QUESTION_GENERATION_PROMPT = """Based on the following content from a project, generate ONE multiple-choice question.

**CRITICAL: DOCUMENT-ONLY QUESTIONS**
⛔ You MUST ONLY create questions that can be answered from the CONTENT provided below
⛔ You MUST NOT use any external knowledge, general facts, or information not in this content
⛔ The correct answer MUST be verifiable from the content provided
⛔ If you cannot create a question from this content, respond with an error

DIFFICULTY LEVEL: {difficulty}
- Easy: Basic recall, definitions, simple facts that are directly stated in the content
- Medium: Understanding concepts and connecting ideas within the content
- Hard: Analysis, synthesis, and nuanced understanding of the content

TOPICS ALREADY ASKED (avoid these): {asked_topics}

PREVIOUS QUESTIONS (DO NOT repeat or ask similar questions):
{previous_questions}

CONTENT (ONLY use information from this content):
{content}

Generate a NEW, UNIQUE question in this EXACT JSON format (no markdown, just pure JSON):
{{
  "question": "The question text (do NOT include code here - use code_snippet instead)",
  "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
  "correct_answer": "A",
  "explanation": "Brief explanation of why this is correct and why others are wrong",
  "topic": "Short topic/concept this question tests (2-4 words)",
  "code_snippet": "```language\\ncode here\\n```"
}}

Requirements:
- Exactly 4 options labeled as the text (not A., B., etc.)
- correct_answer must be one of: "A", "B", "C", "D"
- Only ONE correct answer
- Options should be plausible (no obviously wrong answers)
- **CRITICAL: Question AND correct answer MUST come from the content above - NO external knowledge**
- Explanation should be educational (2-3 sentences explaining WHY the answer is correct based on the content)
- Topic should be a brief descriptor for tracking purposes
- MUST be different from all previous questions listed above
- DO NOT use phrases like "According to the text", "Based on the passage", "The text states" - ask DIRECT questions
- CODE FORMATTING: If the question involves code, put the code in "code_snippet" field with markdown code blocks (```language), NOT in the question text
- If no code is needed, omit the code_snippet field entirely
- RESPOND ONLY WITH THE JSON, NO MARKDOWN FORMATTING"""


# Extended question generation prompt (for questions beyond project content)
EXTENDED_QUESTION_PROMPT = """You are an educational quiz generator creating questions that test deeper understanding.

Based on the project content provided, generate ONE multiple-choice question that:
1. Tests understanding of the TOPIC/SUBJECT matter
2. May go BEYOND what's explicitly stated in the project
3. Should test practical knowledge a student of this material should know
4. Can include related concepts, implications, or real-world applications

The project discusses this main topic: {inferred_topic}

DIFFICULTY LEVEL: {difficulty}
- Easy: Basic concepts related to the topic, definitions, simple applications
- Medium: Connecting ideas, applying knowledge to new situations
- Hard: Analysis, synthesis, edge cases, nuanced understanding, implications

TOPICS ALREADY ASKED (avoid these): {asked_topics}

PREVIOUS QUESTIONS (DO NOT repeat or ask similar questions):
{previous_questions}

CONTENT FROM PROJECT (for context):
{content}

Generate a NEW, UNIQUE question in this EXACT JSON format (no markdown, just pure JSON):
{{
  "type": "multiple_choice",
  "question": "The question text (do NOT include code here - use code_snippet instead)",
  "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
  "correct_answer": "A",
  "explanation": "Brief explanation of why this is correct and why others are wrong",
  "topic": "Short topic/concept this question tests (2-4 words)",
  "code_snippet": "```language\\ncode here\\n```"
}}

Requirements:
- Exactly 4 options labeled as the text (not A., B., etc.)
- correct_answer must be one of: "A", "B", "C", "D"
- Only ONE correct answer
- Options should be plausible (no obviously wrong answers)
- Question can test related knowledge beyond what's explicitly in the content
- Explanation should be educational (2-3 sentences)
- MUST be different from all previous questions
- CODE FORMATTING: If the question involves code, put the code in "code_snippet" field with markdown code blocks (```language), NOT in the question text
- If no code is needed, omit the code_snippet field entirely
- RESPOND ONLY WITH THE JSON, NO MARKDOWN FORMATTING"""


# Code question generation prompt
CODE_QUESTION_PROMPT = """You are a coding quiz generator creating questions that test programming understanding.

Based on the technical content in the project, generate a CODE-BASED question.

QUESTION TYPE: {question_type}
- code_output: Show code, ask "What will be the output of this code?"
- code_fill_blank: Show code with a missing part marked as ___BLANK___, ask to fill it correctly
- code_bug_fix: Show buggy code, ask to identify the issue or fix

DIFFICULTY LEVEL: {difficulty}
- Easy: Simple syntax, basic operations, straightforward logic
- Medium: Functions, loops, conditionals, common patterns
- Hard: Complex logic, edge cases, subtle bugs, advanced concepts

PROGRAMMING LANGUAGE: {language}

TOPICS ALREADY ASKED (avoid these): {asked_topics}

PREVIOUS QUESTIONS (DO NOT repeat):
{previous_questions}

CONTENT FROM PROJECT (for context and topic ideas):
{content}

Generate a question in this EXACT JSON format (no markdown, just pure JSON):
{{
  "type": "{question_type}",
  "question": "The question text (e.g., 'What will be the output of this code?')",
  "code_snippet": "```{language}\\ncode here\\n```",
  "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
  "correct_answer": "A",
  "explanation": "Why this is correct, explaining the code behavior",
  "topic": "Short topic this tests (2-4 words)"
}}

Requirements for each question type:
- code_output: Code must be complete and runnable, options are possible outputs
- code_fill_blank: Code has ___BLANK___ marker, options are code completions
- code_bug_fix: Code has a bug, options describe the bug or fix

General requirements:
- Code should be realistic and educational
- Exactly 4 options
- correct_answer must be "A", "B", "C", or "D"
- Code snippet MUST use triple backticks with language marker
- Explanation should explain the code behavior
- RESPOND ONLY WITH THE JSON, NO MARKDOWN FORMATTING"""


# Topic inference prompt
TOPIC_INFERENCE_PROMPT = """Analyze this content and identify the main topic/subject being studied.
Return ONLY a short topic description (2-5 words). No explanation, just the topic.

Examples of good responses:
- "Python Data Structures"
- "React Component Design"
- "Machine Learning Basics"
- "Database Normalization"
- "HTTP REST APIs"

Content:
{content}

Topic:"""


class QuizService(BaseService):
    """Service for quiz operations.

    Handles quiz creation, question generation using LLM, adaptive difficulty
    adjustment, and result tracking. Uses the RAG pipeline to retrieve
    relevant project content for question generation.
    """

    # === Admin/Creator Methods ===

    def create_quiz(
        self,
        project_id: str,
        user_id: str,
        title: str,
        num_questions: int = 10,
        difficulty_mode: str = 'adaptive',
        time_limit: Optional[int] = None,
        llm_model: Optional[str] = None,
        question_source: str = 'project_only',
        include_code_questions: bool = False
    ) -> Dict[str, Any]:
        """Create a new quiz configuration.

        Args:
            project_id: UUID of the project to generate questions from
            user_id: UUID of the user creating the quiz
            title: Title for the quiz
            num_questions: Number of questions (default 10)
            difficulty_mode: 'adaptive', 'easy', 'medium', or 'hard'
            time_limit: Optional time limit in minutes
            llm_model: Optional LLM model for question generation
            question_source: 'project_only' or 'extended' for broader questions
            include_code_questions: Whether to generate code-based questions

        Returns:
            Dictionary with quiz_id, title, and shareable link

        Raises:
            ValueError: If project doesn't exist or has no content
        """
        self._validate_database_available()

        with self.db_manager.get_session() as session:
            # Verify project exists and has content
            project = session.execute(
                select(Project).where(Project.project_id == project_id)
            ).scalar_one_or_none()

            if not project:
                raise ValueError(f"Project {project_id} not found")

            # Check if project has active sources
            source_count = session.execute(
                select(func.count(CodeFile.source_id))
                .where(CodeFile.project_id == project_id)
                .where(CodeFile.active == True)
            ).scalar()

            if source_count == 0:
                raise ValueError(f"Project {project_id} has no active documents")

            # Validate difficulty mode
            valid_modes = ['adaptive', 'easy', 'medium', 'hard']
            if difficulty_mode not in valid_modes:
                raise ValueError(f"Invalid difficulty_mode. Must be one of: {valid_modes}")

            # Validate question source
            valid_sources = ['project_only', 'extended']
            if question_source not in valid_sources:
                raise ValueError(f"Invalid question_source. Must be one of: {valid_sources}")

            # Create quiz record
            quiz = Quiz(
                id=uuid.uuid4(),
                project_id=project_id,
                user_id=user_id,
                title=title,
                num_questions=num_questions,
                difficulty_mode=difficulty_mode,
                time_limit_minutes=time_limit,
                llm_model=llm_model,
                is_active=True,
                question_source=question_source,
                include_code_questions=include_code_questions
            )

            session.add(quiz)
            session.commit()

            self.logger.info(f"Created quiz {quiz.id}: {title} ({num_questions} questions, source={question_source}, code={include_code_questions})")

            return {
                'quiz_id': str(quiz.id),
                'title': title,
                'link': f"/quiz/{quiz.id}",
                'num_questions': num_questions,
                'difficulty_mode': difficulty_mode,
                'time_limit': time_limit,
                'llm_model': llm_model,
                'question_source': question_source,
                'include_code_questions': include_code_questions
            }

    def get_quiz_results(self, quiz_id: str, user_id: str) -> Dict[str, Any]:
        """Get all attempts for a quiz (creator only).

        Args:
            quiz_id: UUID of the quiz
            user_id: UUID of the requesting user (must be creator)

        Returns:
            Dictionary with quiz info and list of attempts
        """
        self._validate_database_available()

        with self.db_manager.get_session() as session:
            # Get quiz and verify ownership
            quiz = session.execute(
                select(Quiz).where(Quiz.id == quiz_id)
            ).scalar_one_or_none()

            if not quiz:
                raise ValueError(f"Quiz {quiz_id} not found")

            if str(quiz.user_id) != user_id:
                raise PermissionError("Only the quiz creator can view results")

            # Get project name
            project = session.execute(
                select(Project).where(Project.project_id == quiz.project_id)
            ).scalar_one_or_none()

            # Get all attempts
            attempts = session.execute(
                select(QuizAttempt)
                .where(QuizAttempt.quiz_id == quiz_id)
                .order_by(desc(QuizAttempt.started_at))
            ).scalars().all()

            # Calculate statistics
            completed_attempts = [a for a in attempts if a.completed_at is not None]
            scores = [a.score for a in completed_attempts]
            avg_score = sum(scores) / len(scores) if scores else 0
            pass_rate = len([s for s in scores if s / quiz.num_questions >= 0.6]) / len(scores) * 100 if scores else 0

            return {
                'quiz': {
                    'id': str(quiz.id),
                    'title': quiz.title,
                    'project_id': str(quiz.project_id),
                    'project_name': project.name if project else 'Unknown',
                    'num_questions': quiz.num_questions,
                    'difficulty_mode': quiz.difficulty_mode,
                    'time_limit': quiz.time_limit_minutes,
                    'is_active': quiz.is_active,
                    'created_at': quiz.created_at.isoformat()
                },
                'statistics': {
                    'total_attempts': len(attempts),
                    'completed_attempts': len(completed_attempts),
                    'avg_score': round(avg_score, 1),
                    'avg_percentage': round(avg_score / quiz.num_questions * 100, 1) if quiz.num_questions > 0 else 0,
                    'pass_rate': round(pass_rate, 1)
                },
                'attempts': [
                    {
                        'id': str(a.id),
                        'taker_name': a.taker_name,
                        'score': a.score,
                        'total': a.total_questions,
                        'percentage': round(a.score / a.total_questions * 100, 1) if a.total_questions > 0 else 0,
                        'passed': a.score / a.total_questions >= 0.6 if a.total_questions > 0 else False,
                        'started_at': a.started_at.isoformat(),
                        'completed_at': a.completed_at.isoformat() if a.completed_at else None
                    }
                    for a in attempts
                ]
            }

    def list_quizzes(self, user_id: str) -> List[Dict[str, Any]]:
        """List all quizzes created by user.

        Args:
            user_id: UUID of the user

        Returns:
            List of quiz summaries
        """
        self._validate_database_available()

        with self.db_manager.get_session() as session:
            # Get quizzes with attempt counts
            quizzes = session.execute(
                select(Quiz)
                .where(Quiz.user_id == user_id)
                .where(Quiz.is_active == True)
                .order_by(desc(Quiz.created_at))
            ).scalars().all()

            result = []
            for quiz in quizzes:
                # Get project name
                project = session.execute(
                    select(Project).where(Project.project_id == quiz.project_id)
                ).scalar_one_or_none()

                # Count attempts
                attempt_count = session.execute(
                    select(func.count(QuizAttempt.id))
                    .where(QuizAttempt.quiz_id == quiz.id)
                ).scalar()

                result.append({
                    'id': str(quiz.id),
                    'title': quiz.title,
                    'project_id': str(quiz.project_id),
                    'project_name': project.name if project else 'Unknown',
                    'num_questions': quiz.num_questions,
                    'difficulty_mode': quiz.difficulty_mode,
                    'time_limit': quiz.time_limit_minutes,
                    'attempt_count': attempt_count,
                    'link': f"/quiz/{quiz.id}",
                    'created_at': quiz.created_at.isoformat()
                })

            return result

    def delete_quiz(self, quiz_id: str, user_id: str) -> bool:
        """Deactivate a quiz (soft delete).

        Args:
            quiz_id: UUID of the quiz
            user_id: UUID of the user (must be creator)

        Returns:
            True if deleted successfully
        """
        self._validate_database_available()

        with self.db_manager.get_session() as session:
            quiz = session.execute(
                select(Quiz).where(Quiz.id == quiz_id)
            ).scalar_one_or_none()

            if not quiz:
                raise ValueError(f"Quiz {quiz_id} not found")

            if str(quiz.user_id) != user_id:
                raise PermissionError("Only the quiz creator can delete it")

            quiz.is_active = False
            session.commit()

            self.logger.info(f"Deactivated quiz {quiz_id}")
            return True

    # === Public/Test-Taker Methods ===

    def get_quiz_info(self, quiz_id: str) -> Dict[str, Any]:
        """Get public quiz info (for landing page).

        Args:
            quiz_id: UUID of the quiz

        Returns:
            Public quiz information
        """
        self._validate_database_available()

        with self.db_manager.get_session() as session:
            quiz = session.execute(
                select(Quiz).where(Quiz.id == quiz_id)
            ).scalar_one_or_none()

            if not quiz:
                raise ValueError(f"Quiz {quiz_id} not found")

            if not quiz.is_active:
                raise ValueError("This quiz is no longer active")

            return {
                'quiz_id': str(quiz.id),
                'title': quiz.title,
                'num_questions': quiz.num_questions,
                'difficulty_mode': quiz.difficulty_mode,
                'time_limit': quiz.time_limit_minutes,
                'has_time_limit': quiz.time_limit_minutes is not None,
                'question_source': quiz.question_source,
                'include_code_questions': quiz.include_code_questions
            }

    def start_attempt(
        self,
        quiz_id: str,
        taker_name: str,
        taker_email: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start a quiz attempt and generate first question.

        If email is provided and there's an existing incomplete attempt,
        returns that attempt for resumption.

        Args:
            quiz_id: UUID of the quiz
            taker_name: Name of the person taking the quiz
            taker_email: Optional email for session resumption

        Returns:
            Dictionary with attempt_id, first question, and resumed flag
        """
        self._validate_database_available()

        with self.db_manager.get_session() as session:
            # Get quiz
            quiz = session.execute(
                select(Quiz).where(Quiz.id == quiz_id)
            ).scalar_one_or_none()

            if not quiz:
                raise ValueError(f"Quiz {quiz_id} not found")

            if not quiz.is_active:
                raise ValueError("This quiz is no longer active")

            # Check for existing incomplete attempt if email provided
            existing_attempt = None
            if taker_email:
                taker_email = taker_email.strip().lower()
                existing_attempt = session.execute(
                    select(QuizAttempt).where(
                        QuizAttempt.quiz_id == quiz_id,
                        QuizAttempt.taker_email == taker_email,
                        QuizAttempt.completed_at.is_(None)  # Not completed
                    ).order_by(desc(QuizAttempt.started_at))
                ).scalar_one_or_none()

            if existing_attempt:
                # Resume existing attempt
                answers = existing_attempt.answers_json or []
                current_q = answers[-1] if answers else None

                self.logger.info(
                    f"Resuming quiz attempt {existing_attempt.id} for {taker_email}"
                )

                return {
                    'attempt_id': str(existing_attempt.id),
                    'quiz_title': quiz.title,
                    'resumed': True,
                    'question': {
                        'question': current_q['question'] if current_q else '',
                        'options': current_q['options'] if current_q else [],
                    } if current_q and current_q.get('user_answer') is None else None,
                    'question_num': existing_attempt.current_question + 1,
                    'total': existing_attempt.total_questions,
                    'score': existing_attempt.score,
                    'time_limit': quiz.time_limit_minutes,
                    'difficulty': DIFFICULTY_LABELS[existing_attempt.current_difficulty]
                }

            # Determine initial difficulty
            if quiz.difficulty_mode == 'adaptive':
                initial_difficulty = DIFFICULTY_MEDIUM
            elif quiz.difficulty_mode == 'easy':
                initial_difficulty = DIFFICULTY_EASY
            elif quiz.difficulty_mode == 'medium':
                initial_difficulty = DIFFICULTY_MEDIUM
            else:  # hard
                initial_difficulty = DIFFICULTY_HARD

            # Create attempt record
            attempt = QuizAttempt(
                id=uuid.uuid4(),
                quiz_id=quiz_id,
                taker_name=taker_name.strip(),
                taker_email=taker_email,
                score=0,
                total_questions=quiz.num_questions,
                answers_json=[],
                current_question=0,
                current_difficulty=initial_difficulty
            )

            session.add(attempt)
            session.commit()

            # Generate first question
            question = self._generate_question_for_attempt(
                str(quiz.project_id),
                DIFFICULTY_LABELS[initial_difficulty],
                [],
                quiz.llm_model,
                user_id=str(quiz.user_id),  # Log against quiz creator
                question_source=quiz.question_source,
                include_code_questions=quiz.include_code_questions
            )

            # Save the question to answers_json so submit_answer can find it
            attempt.answers_json = [question]
            flag_modified(attempt, 'answers_json')
            session.commit()

            self.logger.info(f"Started quiz attempt {attempt.id} for {taker_name}")

            return {
                'attempt_id': str(attempt.id),
                'quiz_title': quiz.title,
                'resumed': False,
                'question': question,
                'question_num': 1,
                'total': quiz.num_questions,
                'time_limit': quiz.time_limit_minutes,
                'difficulty': DIFFICULTY_LABELS[initial_difficulty]
            }

    def submit_answer(self, attempt_id: str, answer: str) -> Dict[str, Any]:
        """Process answer, adjust difficulty, get next question or results.

        Args:
            attempt_id: UUID of the attempt
            answer: User's answer ('A', 'B', 'C', or 'D')

        Returns:
            Dictionary with correctness and next question or final results
        """
        self._validate_database_available()

        with self.db_manager.get_session() as session:
            # Get attempt
            attempt = session.execute(
                select(QuizAttempt).where(QuizAttempt.id == attempt_id)
            ).scalar_one_or_none()

            if not attempt:
                raise ValueError(f"Attempt {attempt_id} not found")

            if attempt.completed_at is not None:
                raise ValueError("This quiz has already been completed")

            # Get quiz for settings and validate it's still active
            quiz = session.execute(
                select(Quiz).where(Quiz.id == attempt.quiz_id)
            ).scalar_one_or_none()

            if not quiz:
                raise ValueError("Quiz not found")

            if not quiz.is_active:
                raise ValueError("This quiz is no longer available")

            # Get the current question from answers_json
            answers = attempt.answers_json or []

            if attempt.current_question >= len(answers):
                raise ValueError("No pending question to answer")

            current_q = answers[-1] if answers else None
            if current_q is None or current_q.get('user_answer') is not None:
                raise ValueError("No pending question to answer")

            # Normalize answer
            answer = answer.upper().strip()
            if answer not in ['A', 'B', 'C', 'D']:
                raise ValueError("Answer must be A, B, C, or D")

            # Check correctness
            is_correct = answer == current_q['correct_answer']

            # Update answer record
            current_q['user_answer'] = answer
            current_q['correct'] = is_correct

            # Update score
            if is_correct:
                attempt.score += 1

            # Adjust difficulty for adaptive mode
            if quiz.difficulty_mode == 'adaptive':
                if is_correct and attempt.current_difficulty < DIFFICULTY_HARD:
                    attempt.current_difficulty += 1
                elif not is_correct and attempt.current_difficulty > DIFFICULTY_EASY:
                    attempt.current_difficulty -= 1

            # Move to next question
            attempt.current_question += 1
            attempt.answers_json = answers
            flag_modified(attempt, 'answers_json')

            session.commit()

            # Check if quiz is complete
            if attempt.current_question >= attempt.total_questions:
                attempt.completed_at = datetime.utcnow()
                session.commit()

                # Return final results
                return {
                    'correct': is_correct,
                    'explanation': current_q['explanation'],
                    'correct_answer': current_q['correct_answer'],
                    'completed': True,
                    'results': {
                        'score': attempt.score,
                        'total': attempt.total_questions,
                        'percentage': round(attempt.score / attempt.total_questions * 100, 1),
                        'passed': attempt.score / attempt.total_questions >= 0.6,
                        'answers': answers
                    }
                }

            # Generate next question (pass all previous answers to avoid duplicates)
            next_question = self._generate_question_for_attempt(
                str(quiz.project_id),
                DIFFICULTY_LABELS[attempt.current_difficulty],
                answers,  # Full list of previous Q&A for duplicate prevention
                quiz.llm_model,  # Use quiz-specific LLM if configured
                user_id=str(quiz.user_id),  # Log against quiz creator
                question_source=quiz.question_source,
                include_code_questions=quiz.include_code_questions
            )

            # Add next question to answers
            answers.append(next_question)
            attempt.answers_json = answers
            flag_modified(attempt, 'answers_json')
            session.commit()

            # Build next_question response with optional code_snippet
            next_question_response = {
                'question': next_question['question'],
                'options': next_question['options'],
                'question_num': attempt.current_question + 1,
                'total': attempt.total_questions,
                'difficulty': DIFFICULTY_LABELS[attempt.current_difficulty],
                'type': next_question.get('type', 'multiple_choice')
            }
            if next_question.get('code_snippet'):
                next_question_response['code_snippet'] = next_question['code_snippet']

            return {
                'correct': is_correct,
                'explanation': current_q['explanation'],
                'correct_answer': current_q['correct_answer'],
                'completed': False,
                'next_question': next_question_response
            }

    def get_attempt_status(self, attempt_id: str) -> Dict[str, Any]:
        """Get current status of an attempt (for resuming).

        Args:
            attempt_id: UUID of the attempt

        Returns:
            Current attempt status and question if incomplete
        """
        self._validate_database_available()

        with self.db_manager.get_session() as session:
            attempt = session.execute(
                select(QuizAttempt).where(QuizAttempt.id == attempt_id)
            ).scalar_one_or_none()

            if not attempt:
                raise ValueError(f"Attempt {attempt_id} not found")

            quiz = session.execute(
                select(Quiz).where(Quiz.id == attempt.quiz_id)
            ).scalar_one_or_none()

            answers = attempt.answers_json or []

            if attempt.completed_at is not None:
                return {
                    'completed': True,
                    'results': {
                        'score': attempt.score,
                        'total': attempt.total_questions,
                        'percentage': round(attempt.score / attempt.total_questions * 100, 1),
                        'passed': attempt.score / attempt.total_questions >= 0.6,
                        'answers': answers
                    }
                }

            # Get current pending question
            current_q = answers[-1] if answers else None

            return {
                'completed': False,
                'quiz_title': quiz.title,
                'taker_name': attempt.taker_name,
                'question_num': attempt.current_question + 1,
                'total': attempt.total_questions,
                'score': attempt.score,
                'current_question': {
                    'question': current_q['question'],
                    'options': current_q['options'],
                    'difficulty': DIFFICULTY_LABELS[attempt.current_difficulty]
                } if current_q and current_q.get('user_answer') is None else None
            }

    # === Internal Methods ===

    def _get_llm_for_quiz(self, llm_model: Optional[str] = None):
        """Get LLM instance for quiz question generation.

        Args:
            llm_model: Optional model string in format "provider:model" or just "model"
                      Examples: "ollama:llama3.1", "groq:llama-3.1-70b", "llama3.1"

        Returns:
            LLM instance
        """
        if not llm_model:
            # Use default pipeline LLM
            return self.pipeline.get_llm()

        try:
            from ..model import LocalRAGModel

            # Parse provider:model format - we just need the model name
            # LocalRAGModel.set() will auto-detect the provider
            if ':' in llm_model:
                _, model = llm_model.split(':', 1)
            else:
                model = llm_model

            # Get the Ollama host from environment (default to localhost for local dev)
            ollama_host = os.getenv("OLLAMA_HOST", "localhost")

            # Use LocalRAGModel to create the LLM instance
            return LocalRAGModel.set(model_name=model, host=ollama_host)
        except Exception as e:
            self.logger.warning(f"Failed to load LLM '{llm_model}': {e}, falling back to default")
            return self.pipeline.get_llm()

    def _generate_question_for_attempt(
        self,
        project_id: str,
        difficulty: str,
        previous_answers: List[Dict[str, Any]],
        llm_model: Optional[str] = None,
        user_id: Optional[str] = None,
        question_source: str = 'project_only',
        include_code_questions: bool = False
    ) -> Dict[str, Any]:
        """Generate a question using LLM with project content.

        Args:
            project_id: UUID of the project
            difficulty: 'easy', 'medium', or 'hard'
            previous_answers: List of previous question dicts to avoid duplicates
            llm_model: Optional LLM model to use (format: "provider:model" or "model")
            user_id: Optional user ID for query logging
            question_source: 'project_only' or 'extended'
            include_code_questions: Whether to generate code-based questions

        Returns:
            Question dictionary with question, options, correct_answer, explanation, topic
            For code questions, also includes type and code_snippet fields
        """
        # Get project content using retrieval
        content = self._get_project_content(project_id, difficulty)

        if not content:
            raise ValueError("Could not retrieve project content for question generation")

        # Extract topics and question texts from previous answers
        asked_topics = [a.get('topic', '') for a in previous_answers if a.get('topic')]
        previous_questions = [f"- {a.get('question', '')}" for a in previous_answers if a.get('question')]

        # Determine if we should generate a code question
        # Only if enabled and content appears to be technical (40% chance when applicable)
        use_code_question = False
        detected_language = None
        if include_code_questions:
            detected_language = self._detect_programming_language(content)
            if detected_language and random.random() < 0.4:
                use_code_question = True

        # Choose the appropriate prompt
        if use_code_question and detected_language:
            # Generate a code-based question
            code_question_types = [
                QuestionType.CODE_OUTPUT.value,
                QuestionType.CODE_FILL_BLANK.value,
                QuestionType.CODE_BUG_FIX.value
            ]
            question_type = random.choice(code_question_types)

            prompt = CODE_QUESTION_PROMPT.format(
                question_type=question_type,
                difficulty=difficulty.capitalize(),
                language=detected_language,
                asked_topics=', '.join(asked_topics) if asked_topics else 'None',
                previous_questions='\n'.join(previous_questions) if previous_questions else 'None (this is the first question)',
                content=content[:6000]  # Smaller content for code questions
            )
        elif question_source == 'extended':
            # Generate an extended question (beyond project content)
            inferred_topic = self._infer_project_topic(content, llm_model)

            prompt = EXTENDED_QUESTION_PROMPT.format(
                inferred_topic=inferred_topic,
                difficulty=difficulty.capitalize(),
                asked_topics=', '.join(asked_topics) if asked_topics else 'None',
                previous_questions='\n'.join(previous_questions) if previous_questions else 'None (this is the first question)',
                content=content[:8000]
            )
        else:
            # Standard project-only question
            prompt = QUESTION_GENERATION_PROMPT.format(
                difficulty=difficulty.capitalize(),
                asked_topics=', '.join(asked_topics) if asked_topics else 'None',
                previous_questions='\n'.join(previous_questions) if previous_questions else 'None (this is the first question)',
                content=content[:8000]
            )

        # Generate question using LLM (quiz-specific or default)
        try:
            import time
            start_time = time.time()
            llm = self._get_llm_for_quiz(llm_model)
            response = llm.complete(prompt)
            response_text = str(response).strip()
            response_time_ms = int((time.time() - start_time) * 1000)

            # Log query to QueryLogger for metrics
            if hasattr(self.pipeline, '_query_logger') and self.pipeline._query_logger:
                try:
                    from ..observability.token_counter import get_token_counter
                    token_counter = get_token_counter()
                    prompt_tokens = token_counter.count_tokens(prompt)
                    completion_tokens = token_counter.count_tokens(response_text)
                    model_name = llm.model if hasattr(llm, 'model') else (llm_model or 'unknown')

                    query_type = "Code" if use_code_question else ("Extended" if question_source == 'extended' else "Standard")
                    self.pipeline._query_logger.log_query(
                        project_id=project_id,
                        user_id=user_id or "quiz-system",
                        query_text=f"[Quiz Question Generation - {difficulty} - {query_type}]",
                        model_name=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        response_time_ms=response_time_ms
                    )
                except Exception as log_err:
                    self.logger.warning(f"Failed to log quiz query metrics: {log_err}")

            # Parse JSON response
            question_data = self._parse_question_response(response_text)

            # Validate required fields
            required_fields = ['question', 'options', 'correct_answer', 'explanation', 'topic']
            for field in required_fields:
                if field not in question_data:
                    raise ValueError(f"Missing field: {field}")

            if len(question_data['options']) != 4:
                raise ValueError("Must have exactly 4 options")

            if question_data['correct_answer'] not in ['A', 'B', 'C', 'D']:
                raise ValueError("correct_answer must be A, B, C, or D")

            # Ensure type field is present
            if 'type' not in question_data:
                question_data['type'] = QuestionType.MULTIPLE_CHOICE.value

            return question_data

        except Exception as e:
            self.logger.error(f"Error generating question: {e}")
            # Return a fallback question structure
            return self._create_fallback_question(difficulty)

    def _get_project_content(self, project_id: str, difficulty: str) -> str:
        """Retrieve project content for question generation.

        Args:
            project_id: UUID of the project
            difficulty: Difficulty level affects retrieval strategy

        Returns:
            Concatenated content from project
        """
        try:
            # Switch to project context
            self.pipeline.switch_project(project_id, "00000000-0000-0000-0000-000000000001")

            # Get nodes from vector store
            if hasattr(self.pipeline, '_vector_store') and self.pipeline._vector_store:
                nodes = self.pipeline._vector_store.get_nodes_by_project_sql(project_id)

                if nodes:
                    # Sample nodes based on difficulty (harder = more diverse content)
                    import random
                    sample_size = min(len(nodes), 15 if difficulty == 'hard' else 10 if difficulty == 'medium' else 6)
                    sampled_nodes = random.sample(nodes, sample_size)

                    content_parts = []
                    for node in sampled_nodes:
                        if hasattr(node, 'get_content'):
                            content_parts.append(node.get_content())
                        elif hasattr(node, 'text'):
                            content_parts.append(node.text)

                    return "\n\n---\n\n".join(content_parts)

        except Exception as e:
            self.logger.error(f"Error retrieving project content: {e}")

        return ""

    def _detect_programming_language(self, content: str) -> Optional[str]:
        """Detect the primary programming language in the content.

        Args:
            content: Text content to analyze

        Returns:
            Detected language name or None if not technical content
        """
        # Language indicators with their detection patterns
        language_patterns = {
            'python': [
                r'\bdef\s+\w+\s*\(', r'\bclass\s+\w+:', r'\bimport\s+\w+',
                r'\bfrom\s+\w+\s+import', r'print\s*\(', r'\.py\b',
                r'\bpython\b', r'\bpip\b', r'__\w+__'
            ],
            'javascript': [
                r'\bfunction\s+\w+\s*\(', r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*=',
                r'\bconsole\.log\(', r'\brequire\s*\(', r'\.js\b',
                r'\bjavascript\b', r'\bnpm\b', r'\bnode\b', r'=>'
            ],
            'typescript': [
                r'\binterface\s+\w+', r'\btype\s+\w+\s*=', r':\s*string\b',
                r':\s*number\b', r':\s*boolean\b', r'\.ts\b', r'\btypescript\b'
            ],
            'java': [
                r'\bpublic\s+class\b', r'\bprivate\s+\w+\b', r'\bprotected\b',
                r'\bstatic\s+void\s+main\b', r'System\.out\.print', r'\.java\b'
            ],
            'sql': [
                r'\bSELECT\b', r'\bFROM\b', r'\bWHERE\b', r'\bINSERT\b',
                r'\bUPDATE\b', r'\bDELETE\b', r'\bJOIN\b', r'\bCREATE TABLE\b'
            ],
            'rust': [
                r'\bfn\s+\w+\s*\(', r'\blet\s+mut\b', r'\bimpl\s+\w+',
                r'\bstruct\s+\w+', r'\benum\s+\w+', r'\.rs\b', r'\brust\b'
            ],
            'go': [
                r'\bfunc\s+\w+\s*\(', r'\bpackage\s+\w+', r'\bimport\s+"',
                r'\bvar\s+\w+', r'\.go\b', r'\bgolang\b'
            ],
            'c': [
                r'\#include\s*<', r'\bint\s+main\s*\(', r'\bprintf\s*\(',
                r'\bstruct\s+\w+\s*\{', r'\.c\b', r'\.h\b'
            ],
            'cpp': [
                r'\#include\s*<iostream>', r'\bstd::', r'\bcout\s*<<',
                r'\bcin\s*>>', r'\.cpp\b', r'\bc\+\+\b'
            ]
        }

        content_lower = content.lower()
        content_for_regex = content  # Keep original case for some patterns

        language_scores = {}
        for lang, patterns in language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, content_for_regex, re.IGNORECASE)
                score += len(matches)
            if score > 0:
                language_scores[lang] = score

        if not language_scores:
            return None

        # Return the language with the highest score if it meets minimum threshold
        best_lang = max(language_scores.items(), key=lambda x: x[1])
        if best_lang[1] >= 2:  # At least 2 pattern matches required
            return best_lang[0]

        return None

    def _infer_project_topic(self, content: str, llm_model: Optional[str] = None) -> str:
        """Use LLM to infer the main topic from project content.

        Args:
            content: Project content to analyze
            llm_model: Optional LLM model to use

        Returns:
            Short topic description (2-5 words)
        """
        try:
            prompt = TOPIC_INFERENCE_PROMPT.format(content=content[:4000])

            llm = self._get_llm_for_quiz(llm_model)
            response = llm.complete(prompt)
            topic = str(response).strip()

            # Clean up the response - remove quotes, extra punctuation
            topic = topic.strip('"\'')
            topic = re.sub(r'^Topic:\s*', '', topic, flags=re.IGNORECASE)

            # Limit length
            words = topic.split()[:6]
            return ' '.join(words)

        except Exception as e:
            self.logger.warning(f"Failed to infer project topic: {e}")
            return "General Knowledge"

    def _parse_question_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response into a question dictionary.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed question dictionary
        """
        # Try to extract JSON from the response
        # First, try to find JSON block
        json_match = re.search(r'\{[^{}]*"question"[^{}]*\}', response_text, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try parsing the entire response as JSON
        try:
            # Remove markdown code blocks if present
            clean_text = response_text.strip()
            if clean_text.startswith('```'):
                clean_text = re.sub(r'^```(?:json)?\n?', '', clean_text)
                clean_text = re.sub(r'\n?```$', '', clean_text)

            return json.loads(clean_text)
        except json.JSONDecodeError:
            pass

        # If all parsing fails, raise an error
        raise ValueError(f"Could not parse LLM response as JSON: {response_text[:200]}")

    def _create_fallback_question(self, difficulty: str) -> Dict[str, Any]:
        """Create a fallback question when generation fails.

        Args:
            difficulty: The requested difficulty level

        Returns:
            A generic question dictionary
        """
        return {
            'type': QuestionType.MULTIPLE_CHOICE.value,
            'question': 'Based on the content you have studied, which statement is most accurate?',
            'options': [
                'The content covers multiple related topics',
                'The content focuses on a single narrow topic',
                'The content provides no useful information',
                'The content contradicts itself throughout'
            ],
            'correct_answer': 'A',
            'explanation': 'Most educational content covers multiple related topics to provide comprehensive understanding.',
            'topic': 'General Understanding'
        }

    # === Improvement Suggestions ===

    def generate_improvement_suggestions(
        self,
        attempt_id: str
    ) -> Dict[str, Any]:
        """Generate improvement suggestions based on quiz attempt results.

        For 'extended' quizzes: Uses LLM to generate personalized study recommendations.
        For 'project_only' quizzes: Links to specific document sections for review.

        Args:
            attempt_id: UUID of the completed quiz attempt

        Returns:
            Dictionary with suggestions based on question_source type
        """
        self._validate_database_available()

        with self.db_manager.get_session() as session:
            # Get attempt
            attempt = session.execute(
                select(QuizAttempt).where(QuizAttempt.id == attempt_id)
            ).scalar_one_or_none()

            if not attempt:
                raise ValueError(f"Attempt {attempt_id} not found")

            if attempt.completed_at is None:
                raise ValueError("Quiz attempt not yet completed")

            # Get quiz for settings
            quiz = session.execute(
                select(Quiz).where(Quiz.id == attempt.quiz_id)
            ).scalar_one_or_none()

            if not quiz:
                raise ValueError("Quiz not found")

            # Get wrong answers
            answers = attempt.answers_json or []
            wrong_answers = [a for a in answers if not a.get('correct', True)]

            if not wrong_answers:
                return {
                    'type': 'perfect_score',
                    'message': 'Congratulations! You answered all questions correctly.',
                    'suggestions': []
                }

            # Generate suggestions based on question_source
            if quiz.question_source == 'extended':
                return self._generate_llm_suggestions(
                    wrong_answers,
                    quiz.llm_model,
                    str(quiz.project_id),
                    str(quiz.user_id)
                )
            else:  # project_only
                return self._generate_document_suggestions(
                    wrong_answers,
                    str(quiz.project_id)
                )

    def _generate_llm_suggestions(
        self,
        wrong_answers: List[Dict[str, Any]],
        llm_model: Optional[str],
        project_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Generate LLM-powered improvement suggestions for extended quizzes.

        Args:
            wrong_answers: List of incorrectly answered questions
            llm_model: Optional LLM model to use
            project_id: For logging purposes
            user_id: For logging purposes

        Returns:
            Dictionary with LLM-generated suggestions
        """
        # Build context from wrong answers
        wrong_topics = []
        wrong_details = []
        for i, wa in enumerate(wrong_answers, 1):
            topic = wa.get('topic', 'Unknown Topic')
            question = wa.get('question', '')
            explanation = wa.get('explanation', '')
            wrong_topics.append(topic)
            wrong_details.append(
                f"{i}. Topic: {topic}\n"
                f"   Question: {question[:200]}...\n"
                f"   Explanation: {explanation}"
            )

        prompt = f"""Based on the following quiz results, provide personalized study recommendations.

The student answered these questions incorrectly:

{chr(10).join(wrong_details)}

Topics needing improvement: {', '.join(set(wrong_topics))}

Generate 3-5 specific, actionable study recommendations in this JSON format:
{{
  "summary": "Brief overall assessment (1-2 sentences)",
  "weak_areas": ["area1", "area2", ...],
  "suggestions": [
    {{
      "title": "Recommendation title",
      "description": "Detailed actionable advice (2-3 sentences)",
      "priority": "high|medium|low",
      "topics": ["related", "topics"]
    }}
  ],
  "resources": [
    {{
      "type": "concept|practice|reference",
      "title": "Resource or action name",
      "description": "What to study or practice"
    }}
  ]
}}

Focus on:
- Specific concepts the student should review
- Practical exercises to reinforce understanding
- Common misconceptions revealed by wrong answers
- Building from foundational to advanced understanding

RESPOND ONLY WITH JSON, NO MARKDOWN FORMATTING."""

        try:
            import time
            start_time = time.time()
            llm = self._get_llm_for_quiz(llm_model)
            response = llm.complete(prompt)
            response_text = str(response).strip()
            response_time_ms = int((time.time() - start_time) * 1000)

            # Log query metrics
            if hasattr(self.pipeline, '_query_logger') and self.pipeline._query_logger:
                try:
                    from ..observability.token_counter import get_token_counter
                    token_counter = get_token_counter()
                    prompt_tokens = token_counter.count_tokens(prompt)
                    completion_tokens = token_counter.count_tokens(response_text)
                    model_name = llm.model if hasattr(llm, 'model') else (llm_model or 'unknown')

                    self.pipeline._query_logger.log_query(
                        project_id=project_id,
                        user_id=user_id,
                        query_text="[Quiz Improvement Suggestions Generation]",
                        model_name=model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        response_time_ms=response_time_ms
                    )
                except Exception as log_err:
                    self.logger.warning(f"Failed to log suggestion query metrics: {log_err}")

            # Parse response
            suggestions_data = self._parse_suggestion_response(response_text)

            return {
                'type': 'llm_generated',
                'wrong_count': len(wrong_answers),
                'total_topics': list(set(t.get('topic', 'Unknown') for t in wrong_answers)),
                **suggestions_data
            }

        except Exception as e:
            self.logger.error(f"Error generating LLM suggestions: {e}")
            # Return basic suggestions on error
            topics = list(set(wa.get('topic', 'Unknown Topic') for wa in wrong_answers))
            return {
                'type': 'llm_generated',
                'wrong_count': len(wrong_answers),
                'summary': f"You missed {len(wrong_answers)} questions. Focus on reviewing the topics below.",
                'weak_areas': topics,
                'suggestions': [
                    {
                        'title': f'Review: {topic}',
                        'description': 'Review this topic thoroughly and practice related exercises.',
                        'priority': 'high',
                        'topics': [topic]
                    }
                    for topic in topics[:3]
                ],
                'resources': []
            }

    def _generate_document_suggestions(
        self,
        wrong_answers: List[Dict[str, Any]],
        project_id: str
    ) -> Dict[str, Any]:
        """Generate document-linked suggestions for project_only quizzes.

        Finds relevant document sections that cover the topics the student
        struggled with.

        Args:
            wrong_answers: List of incorrectly answered questions
            project_id: UUID of the project

        Returns:
            Dictionary with document section links
        """
        # Extract topics from wrong answers
        topics = list(set(wa.get('topic', 'Unknown Topic') for wa in wrong_answers))

        # Get document sources for the project
        document_sections = []

        try:
            with self.db_manager.get_session() as session:
                # Get active project sources
                sources = session.execute(
                    select(CodeFile)
                    .where(CodeFile.project_id == project_id)
                    .where(CodeFile.active == True)
                ).scalars().all()

                source_info = {str(s.source_id): s.filename for s in sources}

            # Switch to project context for retrieval
            self.pipeline.switch_project(project_id, "00000000-0000-0000-0000-000000000001")

            # For each wrong answer, find relevant document sections
            for wa in wrong_answers:
                topic = wa.get('topic', '')
                question = wa.get('question', '')

                # Use the question text to find relevant chunks
                search_query = f"{topic}: {question[:100]}"

                if hasattr(self.pipeline, '_vector_store') and self.pipeline._vector_store:
                    # Search for relevant nodes
                    try:
                        from llama_index.core import VectorStoreIndex
                        from llama_index.core.retrievers import VectorIndexRetriever

                        # Get a few relevant nodes
                        nodes = self.pipeline._vector_store.get_nodes_by_project_sql(
                            project_id, limit=100
                        )

                        if nodes:
                            # Create a quick index for semantic search
                            index = VectorStoreIndex(nodes=nodes)
                            retriever = VectorIndexRetriever(
                                index=index,
                                similarity_top_k=2
                            )
                            retrieved = retriever.retrieve(search_query)

                            for node_with_score in retrieved:
                                node = node_with_score.node
                                metadata = node.metadata or {}
                                source_id = metadata.get('source_id', '')
                                filename = source_info.get(source_id, metadata.get('filename', 'Unknown'))

                                # Get content preview
                                content = node.get_content() if hasattr(node, 'get_content') else node.text
                                preview = content[:200] + '...' if len(content) > 200 else content

                                document_sections.append({
                                    'topic': topic,
                                    'source_id': source_id,
                                    'filename': filename,
                                    'preview': preview,
                                    'relevance_score': round(node_with_score.score or 0, 3)
                                })

                    except Exception as retrieve_err:
                        self.logger.warning(f"Error retrieving sections for topic '{topic}': {retrieve_err}")

        except Exception as e:
            self.logger.error(f"Error generating document suggestions: {e}")

        # Deduplicate and sort by relevance
        seen = set()
        unique_sections = []
        for section in sorted(document_sections, key=lambda x: x.get('relevance_score', 0), reverse=True):
            key = (section['source_id'], section['preview'][:50])
            if key not in seen:
                seen.add(key)
                unique_sections.append(section)

        # Group by topic
        topics_with_sections = {}
        for section in unique_sections:
            topic = section['topic']
            if topic not in topics_with_sections:
                topics_with_sections[topic] = []
            topics_with_sections[topic].append(section)

        return {
            'type': 'document_linked',
            'wrong_count': len(wrong_answers),
            'summary': f"You missed {len(wrong_answers)} questions across {len(topics)} topics. Review the suggested document sections below.",
            'weak_areas': topics,
            'sections': [
                {
                    'topic': topic,
                    'documents': sections[:2]  # Max 2 sections per topic
                }
                for topic, sections in topics_with_sections.items()
            ],
            'total_sections': len(unique_sections)
        }

    def _parse_suggestion_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM suggestion response into structured format.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed suggestion dictionary
        """
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try parsing the entire response
        try:
            clean_text = response_text.strip()
            if clean_text.startswith('```'):
                clean_text = re.sub(r'^```(?:json)?\n?', '', clean_text)
                clean_text = re.sub(r'\n?```$', '', clean_text)
            return json.loads(clean_text)
        except json.JSONDecodeError:
            pass

        # Return basic structure on parse failure
        return {
            'summary': 'Review the topics where you made mistakes.',
            'weak_areas': [],
            'suggestions': [],
            'resources': []
        }
