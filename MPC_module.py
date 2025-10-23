import warnings
import numpy as np
import do_mpc
import casadi as ca
from dataclasses import dataclass
import sys, os



# ==========================
# 1. Suppress unnecessary warnings
# ==========================
# Filter out UserWarnings and DeprecationWarnings to keep console output clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



# ==========================
# 2. Helper class to hide solver prints
# ==========================
class HiddenPrints:
    """
    Context manager to suppress console output from external solvers.
    Redirects stdout to /dev/null (or equivalent) during solver execution.
    
    Usage:
        with HiddenPrints():
            # Code here won't print to console
    """
    def __enter__(self):
        # Save the original stdout reference
        self._original_stdout = sys.stdout
        # Redirect stdout to null device (discards all output)
        sys.stdout = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the null device file handle
        sys.stdout.close()
        # Restore original stdout so normal printing works again
        sys.stdout = self._original_stdout



# ==========================
# 3. Data structure for questions
# ==========================
@dataclass
class QuestionConfig:
    """
    Data class representing an interview question with metadata.
    
    Attributes:
        question_id: Unique identifier for the question
        question_text: The actual question string
        difficulty: Difficulty level ("easy", "medium", "hard")
        base_time: Expected baseline duration in seconds
        category: Question category (e.g., "intro", "technical", "behavioral")
    """
    question_id: int
    question_text: str
    difficulty: str        # "easy", "medium", "hard"
    base_time: float       # Base time expected in seconds
    category: str          # Category, e.g., intro, technical, behavioural, teamwork



# ==========================
# 4. MPC Interview Controller (FIXED VERSION)
# ==========================
class MPCInterviewController:
    """
    Model Predictive Control (MPC) based controller for adaptive interview time allocation.
    
    This controller uses MPC to dynamically allocate time for interview questions based on:
    - Question difficulty and category
    - Remaining session time
    - Historical performance (learning from past responses)
    - Look-ahead optimization over a prediction horizon
    """
    
    def __init__(self, questions, total_session_time=2700):
        """
        Initialize the MPC Interview Controller.
        
        Args:
            questions: List of QuestionConfig objects representing the interview questions
            total_session_time: Total time available for the interview session (default: 2700s = 45 min)
        """
        self.questions = questions
        self.total_session_time = total_session_time
        
        # Dictionary to store performance history for each question category
        # Format: {category: [{'allocated': float, 'actual': float, 'base': float}, ...]}
        self.category_history = {}
        
        # Dictionary to store smoothed category performance factors
        # These factors adjust predictions based on historical performance
        self.prev_category_factors = {}
        
        # Initialize category history for each unique category in questions
        for q in questions:
            if q.category not in self.category_history:
                self.category_history[q.category] = []
        
        # Setup the MPC model (system dynamics)
        self.mpc_model = self.setup_mpc_model()


    def setup_mpc_model(self):
        """
        Define the discrete-time MPC model with state and control variables.
        
        Returns:
            do_mpc.model.Model: Configured model with states and dynamics
            
        State Variables:
            - remaining_time: Time left in the interview session
            - question_idx: Current question index
            
        Control Variable:
            - time_alloc: Time allocated to current question
            
        Dynamics:
            - remaining_time decreases by allocated time
            - question_idx increments by 1 after each question
        """
        # Create discrete-time model (suitable for sequential decision-making)
        model = do_mpc.model.Model('discrete')


        # Define state variables with shape (1,1) for scalar values
        # State 1: Remaining time in the session
        remaining_time = model.set_variable(var_type='_x', var_name='remaining_time', shape=(1,1))
        
        # State 2: Current question index (tracks progress through interview)
        question_idx = model.set_variable(var_type='_x', var_name='question_idx', shape=(1,1))


        # Define control variable (decision variable the MPC optimizes)
        # Control: Time to allocate for the current question
        time_alloc = model.set_variable(var_type='_u', var_name='time_alloc', shape=(1,1))


        # Define system dynamics (how states evolve based on control input)
        # After allocating time_alloc, remaining time decreases
        model.set_rhs('remaining_time', remaining_time - time_alloc)
        
        # After answering a question, move to next question
        model.set_rhs('question_idx', question_idx + 1)


        # Finalize model configuration
        model.setup()
        return model


    def optimize_allocation(self, current_question_idx, remaining_time, horizon=3):
        """
        Optimize time allocation for current question using MPC.
        
        FIXED: The objective function now correctly considers ONLY the current question
        at each MPC stage, not all questions in the horizon. This ensures the control
        variable is optimized for the immediate decision, which is the proper MPC approach.
        
        This method:
        1. Creates a new MPC controller instance
        2. Configures the prediction horizon and constraints
        3. Sets up the cost function (minimize deviation from predicted duration)
        4. Solves the optimization problem
        5. Returns the optimal time allocation
        
        Args:
            current_question_idx: Index of current question in the question list
            remaining_time: Time remaining in the interview session (seconds)
            horizon: Prediction horizon (how many questions ahead to consider)
            
        Returns:
            float: Optimal time allocation for current question (seconds)
        """
        # Create MPC controller using the predefined model
        mpc = do_mpc.controller.MPC(self.mpc_model)


        # Calculate effective horizon (can't look past end of question list)
        n_horizon = min(horizon, len(self.questions) - current_question_idx)
        
        # If no questions remain, return default fallback allocation
        if n_horizon <= 0:
            return 60  # fallback allocation


        # Configure IPOPT solver options to suppress verbose console output
        suppress_ipopt = {
            'ipopt.print_level': 0,    # No iteration output
            'ipopt.sb': 'yes',         # Suppress banner
            'print_time': 0            # Don't print timing information
        }
        
        # Set MPC parameters
        mpc.set_param(
            n_horizon=n_horizon,              # Number of steps to predict ahead
            t_step=1,                         # Time step (1 = one question per step)
            store_full_solution=True,         # Store complete optimization trajectory
            n_robust=1,                       # Robust horizon (1 = deterministic MPC)
            nlpsol_opts=suppress_ipopt        # Pass solver options to IPOPT
        )
        
        # Get reference to control variable for use in objective
        time_alloc = self.mpc_model.u['time_alloc']
        
        # Get CURRENT question only (not all questions in horizon)
        current_question = self.questions[current_question_idx]
        
        # Predict expected duration for CURRENT question based on its attributes and history
        predicted_duration = self.predict_duration_simple(current_question)
        
        # Define stage cost: quadratic penalty for deviation from predicted duration
        # This penalizes allocations that differ from what the model expects
        # Formula: (allocated_time - predicted_time)^2
        lterm = (time_alloc - ca.SX([predicted_duration])) ** 2
        
        # Terminal cost (cost at end of horizon) - not used in this application
        mterm = ca.SX(0)
        

        # Set objective function in the MPC controller
        mpc.set_objective(mterm=mterm, lterm=lterm)


        # Set control constraints (minimum and maximum time per question)
        # Lower bound: At least 30 seconds per question
        mpc.bounds['lower', '_u', 'time_alloc'] = np.array([[30]])
        # Upper bound: At most 300 seconds (5 minutes) per question
        mpc.bounds['upper', '_u', 'time_alloc'] = np.array([[300]])
        
        # Finalize MPC setup (builds optimization problem)
        mpc.setup()


        # Prepare initial state vector [remaining_time, current_question_idx]
        x0 = np.array([remaining_time, current_question_idx]).flatten()
        
        # Set initial state in MPC controller
        mpc.x0 = x0
        
        # Initialize guess for optimization (warm start)
        mpc.set_initial_guess()


        # Solve optimization problem (suppress output using context manager)
        with HiddenPrints():
            try:
                # Compute optimal control input (time allocation)
                u0 = mpc.make_step(x0)
                return float(u0.item())  # Extract scalar value and return
            except Exception as e:
                # If optimization fails, return safe fallback allocation
                # Use minimum of remaining time and 120s (2 minutes)
                return float(min(remaining_time, 120))


    def predict_duration_simple(self, question, velocity_factor=1.0):
        """
        Predict expected duration for a question using multiple factors.
        
        Prediction formula:
            duration = base_time × difficulty_factor × velocity_factor × category_factor
        
        Args:
            question: QuestionConfig object
            velocity_factor: Global scaling factor for response speed (default: 1.0)
            
        Returns:
            float: Predicted duration in seconds
        """
        # Define difficulty multipliers
        DIFFICULTY_FACTORS = {
            "easy": 0.9,      # Easy questions take 90% of base time
            "medium": 1.0,    # Medium questions take 100% of base time
            "hard": 1.2       # Hard questions take 120% of base time
        }
        
        # Get base time from question config
        base = question.base_time
        
        # Get difficulty multiplier (default to 1.0 if unknown difficulty)
        difficulty = DIFFICULTY_FACTORS.get(question.difficulty, 1.0)
        
        # Get category-specific factor based on historical performance
        category = self.compute_category_factor(question.category)
        
        # Compute final prediction
        return base * difficulty * velocity_factor * category


    def compute_category_factor(self, category):
        """
        Compute adaptive category factor based on historical performance.
        
        Uses exponential smoothing to combine:
        - Recent performance (actual vs. predicted time ratio)
        - Previous smoothed factor (for stability)
        
        Args:
            category: Question category string
            
        Returns:
            float: Category adjustment factor (1.0 = no adjustment)
        """
        DEFAULT_CATEGORY_FACTOR = 1.0  # Neutral factor (no adjustment)
        SMOOTHING_ALPHA = 0.7          # Weight for new data (0.7 = 70% new, 30% old)
        
        # Get performance history for this category
        history = self.category_history.get(category, [])
        
        # If no history exists, return default factor
        if not history:
            return DEFAULT_CATEGORY_FACTOR
        
        # Calculate average actual time taken for this category
        avg_actual = sum(h['actual'] for h in history) / len(history)
        
        # Calculate average base time for this category
        avg_base = sum(h['base'] for h in history) / len(history)
        
        # Compute raw factor: ratio of actual to base time
        # If ratio > 1.0: category takes longer than expected
        # If ratio < 1.0: category takes less than expected
        factor = avg_actual / avg_base if avg_base > 0 else DEFAULT_CATEGORY_FACTOR
        
        # Get previous smoothed factor (default to 1.0 if first time)
        prev = self.prev_category_factors.get(category, DEFAULT_CATEGORY_FACTOR)
        
        # Apply exponential smoothing: weighted average of new and old factors
        # This prevents sudden jumps and makes predictions more stable
        smoothed = SMOOTHING_ALPHA * factor + (1 - SMOOTHING_ALPHA) * prev
        
        # Store smoothed factor for next iteration
        self.prev_category_factors[category] = smoothed
        
        return smoothed


    def update_category_history(self, question, allocated, actual):
        """
        Record question performance data for future learning.
        
        This method stores:
        - Allocated time (what MPC recommended)
        - Actual time (what was actually used)
        - Base time (question's baseline)
        
        This data is used by compute_category_factor() to improve predictions.
        
        Args:
            question: QuestionConfig object
            allocated: Time allocated by MPC (seconds)
            actual: Actual time taken (seconds)
        """
        self.category_history[question.category].append({
            'allocated': allocated,  # MPC's recommendation
            'actual': actual,        # What actually happened
            'base': question.base_time  # Question's baseline
        })


    def print_status(self):
        """
        Display learning status showing how the controller has adapted.
        
        Prints:
        - Number of questions answered per category
        - Current category adjustment factor
        """
        print("\nCategory Learning Status:")
        for category, history in self.category_history.items():
            if history:
                # Get current smoothed factor
                factor = self.prev_category_factors.get(category, 1.0)
                print(f"  {category}: {len(history)} questions, factor = {factor:.2f}")
            else:
                print(f"  {category}: no data yet")



# ==========================
# 5. Demo: Run MPC interview session
# ==========================
if __name__ == "__main__":
    # Define interview question set with metadata
    questions = [
        QuestionConfig(1, "Tell me about yourself", "easy", 90, "intro"),
        QuestionConfig(2, "Walk through your resume", "easy", 90, "resume-based"),
        QuestionConfig(3, "Explain supervised vs unsupervised learning", "hard", 300, "technical"),
        QuestionConfig(4, "Describe your ML research project", "medium", 180, "technical"),
        QuestionConfig(5, "Give an example of teamwork", "medium", 120, "behavioral"),
        QuestionConfig(6, "How do you handle conflicts?", "medium", 120, "behavioral"),
        QuestionConfig(7, "Why do you want this role?", "easy", 90, "closing"),
    ]


    # Initialize controller with questions and total session time
    controller = MPCInterviewController(questions, total_session_time=9000)
    
    # Set initial session time (different from total_session_time to show flexibility)
    remaining_time = 6000  # 100 minutes
    current_idx = 0  # Start at first question


    # Print session header
    print("=== MPC Interview Time Allocation Demo (FIXED VERSION) ===")
    print(f"Total session time: {remaining_time} seconds\n")
    print("CHANGE: Objective function now evaluates ONLY the current question")
    print("at each stage, resulting in proper MPC behavior.\n")


    # Main interview loop
    while current_idx < len(questions) and remaining_time > 0:
        # Get current question
        question = questions[current_idx]
        
        # Use MPC to compute optimal time allocation
        # horizon=3 means MPC looks ahead 3 questions when making decision
        allocation = controller.optimize_allocation(current_idx, remaining_time, horizon=3)


        # Calculate expected time for comparison
        expected = controller.predict_duration_simple(question)
        
        # Display question and allocation
        print(f"Q{current_idx + 1} ({question.category}, {question.difficulty}): {question.question_text}")
        print(f"  → Expected: {expected:.1f}s (base={question.base_time}s)")
        print(f"  → Allocated: {allocation:.1f}s")


        # Simulate actual answer time with random variation (80%-120% of allocation)
        # In real application, this would be actual measured time
        actual_time = allocation * (0.8 + 0.4 * np.random.random())
        print(f"  → Actual: {actual_time:.1f}s")


        # Check if answered faster than allocated (time saved)
        if actual_time < allocation:
            print(f"  ✅ Answered early, saved {allocation - actual_time:.1f}s")


        # Update learning: record performance for this category
        controller.update_category_history(question, allocation, actual_time)
        
        # Update remaining time (prevent negative values)
        remaining_time = max(0, remaining_time - actual_time)
        
        # Move to next question
        current_idx += 1


        # Display remaining time
        print(f"  → Remaining session time: {remaining_time:.1f}s\n")


        # Check if session time is exhausted
        if remaining_time <= 0:
            print("⚠️ Session time exhausted!")
            break


    # Print final summary showing what the controller learned
    print("=== Interview Complete ===")
    controller.print_status()
    
    
