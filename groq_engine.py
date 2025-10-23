"""
groq engine module for question presentation and answer evaluation.
"""


# class groqEngine:
class GROQEngine:
    """
    Stub for Groq-based interview engine.
    
    In a real implementation, this would integrate with GROQ API
    or similar LLM service to generate questions, evaluate answers,
    and provide feedback.
    """
    
    def __init__(self):
        self.question_count = 0

    def present_question(self, question_config):
        """
        Present a question to the candidate.
        
        Args:
            question_config: QuestionConfig object containing question details
            
        Returns:
            str: Formatted question text
        """
        self.question_count += 1
        
        print(f"\n{'─'*60}")
        print(f"Question {self.question_count} ({question_config.category.upper()})")
        print(f"{'─'*60}")
        print(f"Difficulty: {question_config.difficulty.capitalize()}")
        print(f"\n{question_config.question_text}\n")
        
        return question_config.question_text

    def evaluate_answer(self, answer, question_config):
        """
        Evaluate candidate's answer (stub implementation).
               
        Args:
            answer: Candidate's answer text
            question_config: The question that was answered
            
        Returns:
            str: Evaluation feedback
        """
        # Simple length-based feedback (replace with actual groq evaluation)
        word_count = len(answer.split())
        
        if word_count < 10:
            feedback = "Response is quite brief. Consider providing more detail."
        elif word_count < 30:
            feedback = "Good attempt. The response covers the basics."
        else:
            feedback = "Excellent detailed response!"
        
        print(f"\nFeedback: {feedback}")
        print(f"   Words: {word_count}")
        
        return feedback

    def handle_timeout(self):
        """Handle timeout event when candidate runs out of time."""
        print(f"\nTime's up for this question!")
        print("   Let's move to the next question to stay on schedule.")

    def handle_inactivity(self):
        """Handle inactivity event when candidate stops responding."""
        print(f"\nNo activity detected for 10 seconds!")
        print("   Moving to next question to keep the interview on track.")

    def generate_closing(self, total_questions, session_time):
        """
        Generate closing remarks for the interview.
        
        Args:
            total_questions: Number of questions answered
            session_time: Total time spent in interview
        """
        print(f"\n{'='*60}")
        print("Interview Session Complete!")
        print(f"{'='*60}")
        print(f"Questions Answered: {total_questions}")
        print(f"Total Time: {session_time//60} minutes {session_time%60} seconds")
        print(f"\nThank you for your time!")
        print(f"{'='*60}\n")
