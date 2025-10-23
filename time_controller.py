"""
Time controller module with MPC integration for dynamic time allocation.
"""
import time
import threading
# from MPC_module import MPCInterviewController, HiddenPrints
from MPC_module import MPCInterviewController, HiddenPrints



class TimeController:
    """
    Enhanced time controller that integrates MPC for dynamic time allocation.
    
    Features:
    - Session-level time tracking (total interview duration)
    - Question-level time tracking (per-question time limits)
    - MPC-based adaptive time allocation
    - Thread-based timeout handling
    - Inactivity detection (auto-skip after threshold seconds)
    """
    
    def __init__(self, questions, session_duration=2700, inactivity_threshold=10):
        """
        Initialize the time controller with MPC integration.
        
        Args:
            questions: List of QuestionConfig objects
            session_duration: Total interview session time in seconds (default: 2700s = 45 min)
            inactivity_threshold: Seconds of inactivity before auto-skip (default: 10)
        """
        self.session_duration = session_duration
        self.start_time = None
        self.question_start_time = None
        self.question_timer = None
        self.remaining_time = session_duration
        self.current_question_idx = 0
        
        # Initialize MPC controller for adaptive time allocation
        self.mpc_controller = MPCInterviewController(
            questions=questions,
            total_session_time=session_duration
        )
        
        # Store questions for reference
        self.questions = questions
        
        # Track timeout events
        self.timeout_occurred = threading.Event()
        self.stop_timer = threading.Event()
        
        # Track inactivity events
        self.inactivity_occurred = threading.Event()
        self.stop_inactivity_timer = threading.Event()
        self.inactivity_timer = None
        self.last_activity_time = None
        self.INACTIVITY_THRESHOLD = inactivity_threshold

    def start_session(self):
        """Start the interview session timer."""
        self.start_time = time.time()
        self.remaining_time = self.session_duration
        print(f"\n{'='*60}")
        print(f"Interview Session Started")
        print(f" Total Duration: {self.session_duration//60} minutes ({self.session_duration}s)")
        print(f" Total Questions: {len(self.questions)}")
        print(f" Auto-skip: Questions skip after {self.INACTIVITY_THRESHOLD}s of no input")
        print(f"{'='*60}\n")

    def get_remaining_time(self):
        """
        Calculate remaining time in the session.
        
        Returns:
            float: Remaining time in seconds (minimum 0)
        """
        if self.start_time is None:
            return self.session_duration
        
        elapsed = time.time() - self.start_time
        self.remaining_time = max(0, self.session_duration - elapsed)
        return self.remaining_time

    def get_mpc_allocation(self):
        """
        Get MPC-optimized time allocation for current question.
        
        Uses Model Predictive Control to dynamically allocate time based on:
        - Current question difficulty and category
        - Remaining session time
        - Historical performance data
        - Look-ahead optimization
        
        Returns:
            float: Allocated time for current question in seconds
        """
        remaining = self.get_remaining_time()
        
        # Use MPC to optimize allocation with 3-step lookahead
        with HiddenPrints():
            allocation = self.mpc_controller.optimize_allocation(
                current_question_idx=self.current_question_idx,
                remaining_time=remaining,
                horizon=3  # Look ahead 3 questions
            )
        
        return allocation

    def start_question_timer(self, limit_seconds, on_timeout_callback):
        """
        Start a timer for the current question with timeout handling.
        
        Uses threading to allow non-blocking timeout detection while
        the candidate is answering the question.
        
        Args:
            limit_seconds: Maximum time allowed for this question
            on_timeout_callback: Function to call when time expires
        """
        # Cancel any existing timer
        self.stop_timer.set()
        if self.question_timer and self.question_timer.is_alive():
            self.question_timer.join(timeout=0.1)
        
        # Reset flags
        self.stop_timer.clear()
        self.timeout_occurred.clear()
        
        # Record question start time
        self.question_start_time = time.time()
        
        def countdown():
            """Timer thread function that monitors timeout."""
            # Wait for either timeout or stop signal
            self.stop_timer.wait(timeout=limit_seconds)
            
            # Check if we should trigger timeout
            if not self.stop_timer.is_set() and self.get_remaining_time() > 0:
                self.timeout_occurred.set()
                on_timeout_callback()
        
        # Start timer thread
        self.question_timer = threading.Thread(target=countdown, daemon=True)
        self.question_timer.start()

    def start_inactivity_monitor(self, on_inactivity_callback):
        """
        Start monitoring for user inactivity.
        
        If the user doesn't provide any input for INACTIVITY_THRESHOLD seconds,
        automatically move to the next question.
        
        Args:
            on_inactivity_callback: Function to call when inactivity is detected
        """
        # Cancel any existing inactivity timer
        self.stop_inactivity_timer.set()
        if self.inactivity_timer and self.inactivity_timer.is_alive():
            self.inactivity_timer.join(timeout=0.1)
        
        # Reset flags
        self.stop_inactivity_timer.clear()
        self.inactivity_occurred.clear()
        
        # Record initial activity time
        self.last_activity_time = time.time()
        
        def monitor_inactivity():
            """Thread that monitors for periods of inactivity."""
            while not self.stop_inactivity_timer.is_set():
                # Check how long since last activity
                time_since_activity = time.time() - self.last_activity_time
                
                # If inactivity threshold exceeded, trigger callback
                if time_since_activity >= self.INACTIVITY_THRESHOLD:
                    if not self.stop_inactivity_timer.is_set():
                        self.inactivity_occurred.set()
                        on_inactivity_callback()
                        break
                
                # Check every 0.5 seconds
                time.sleep(0.5)
        
        # Start monitoring thread
        self.inactivity_timer = threading.Thread(target=monitor_inactivity, daemon=True)
        self.inactivity_timer.start()

    def record_activity(self):
        """
        Record that user activity has occurred.
        
        Call this whenever the user provides input to reset the inactivity timer.
        """
        self.last_activity_time = time.time()

    def stop_inactivity_monitor(self):
        """
        Stop the inactivity monitoring thread.
        
        Call this when the question is complete.
        """
        self.stop_inactivity_timer.set()
        if self.inactivity_timer and self.inactivity_timer.is_alive():
            self.inactivity_timer.join(timeout=0.1)

    def stop_question_timer(self):
        """
        Stop the current question timer and calculate actual time spent.
        
        Returns:
            float: Actual time spent on the question in seconds
        """
        # Signal timer to stop
        self.stop_timer.set()
        
        # Stop inactivity monitor
        self.stop_inactivity_monitor()
        
        # Calculate actual time spent
        if self.question_start_time:
            actual_time = time.time() - self.question_start_time
            self.question_start_time = None
            return actual_time
        return 0

    def update_mpc_with_performance(self, question, allocated_time, actual_time):
        """
        Update MPC controller with performance data for learning.
        
        This allows the MPC to adapt its predictions based on how long
        questions actually take compared to allocations.
        
        Args:
            question: QuestionConfig object
            allocated_time: Time that was allocated by MPC
            actual_time: Actual time spent answering
        """
        self.mpc_controller.update_category_history(
            question=question,
            allocated=allocated_time,
            actual=actual_time
        )

    def move_to_next_question(self):
        """Increment the question index."""
        self.current_question_idx += 1

    def is_time_up(self):
        """
        Check if session time has expired.
        
        Returns:
            bool: True if no time remains, False otherwise
        """
        return self.get_remaining_time() <= 0

    def has_more_questions(self):
        """
        Check if there are more questions to ask.
        
        Returns:
            bool: True if more questions remain, False otherwise
        """
        return self.current_question_idx < len(self.questions)

    def get_current_question(self):
        """
        Get the current question object.
        
        Returns:
            QuestionConfig: Current question, or None if no more questions
        """
        if self.has_more_questions():
            return self.questions[self.current_question_idx]
        return None

    def print_mpc_learning_status(self):
        """Display what the MPC controller has learned during the session."""
        print(f"\n{'='*60}")
        print("MPC Learning Summary")
        print(f"{'='*60}")
        self.mpc_controller.print_status()
        print(f"{'='*60}\n")
