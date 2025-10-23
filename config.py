"""
Configuration and question definitions for the MPC Interview System.
"""
# from MPC_module import QuestionConfig
from MPC_module import QuestionConfig


# Interview question configurations
QUESTIONS = [
    QuestionConfig(1, "Tell me about yourself", "easy", 90, "intro"),
    QuestionConfig(2, "Walk through your resume", "easy", 90, "resume-based"),
    QuestionConfig(3, "Explain supervised vs unsupervised learning", "hard", 300, "technical"),
    QuestionConfig(4, "Describe your ML research project", "medium", 180, "technical"),
    QuestionConfig(5, "Give an example of teamwork", "medium", 120, "behavioral"),
    QuestionConfig(6, "How do you handle conflicts?", "medium", 120, "behavioral"),
    QuestionConfig(7, "Why do you want this role?", "easy", 90, "closing"),
]

# Session configuration
DEFAULT_SESSION_DURATION = 600  # 10 minutes for testing 
INACTIVITY_THRESHOLD = 10  # seconds
