import warnings
import numpy as np
import do_mpc
import casadi as ca
from dataclasses import dataclass
import sys, os

# ==========================
# 1. Suppress unnecessary warnings
# ==========================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ==========================
# 2. Helper class to hide solver prints
# ==========================
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# ==========================
# 3. Data structure for questions
# ==========================
@dataclass
class QuestionConfig:
    question_id: int
    question_text: str
    difficulty: str         # "easy", "medium", "hard"
    base_time: float        # Base time in seconds
    category: str           # e.g., intro, technical, behavioral

# ==========================
# 4. MPC Interview Controller, velocity-adaptive
# ==========================
class MPCInterviewController:
    def __init__(self, questions, total_session_time=2700):
        self.questions = questions
        self.total_session_time = total_session_time

        self.category_history = {}
        self.prev_category_factors = {}
        for q in questions:
            if q.category not in self.category_history:
                self.category_history[q.category] = []
        self.mpc_model = self.setup_mpc_model()

    def setup_mpc_model(self):
        model = do_mpc.model.Model('discrete')
        remaining_time = model.set_variable(var_type='_x', var_name='remaining_time', shape=(1,1))
        question_idx = model.set_variable(var_type='_x', var_name='question_idx', shape=(1,1))
        time_alloc = model.set_variable(var_type='_u', var_name='time_alloc', shape=(1,1))
        model.set_rhs('remaining_time', remaining_time - time_alloc)
        model.set_rhs('question_idx', question_idx + 1)
        model.setup()
        return model

    def optimize_allocation(self, current_question_idx, remaining_time, horizon=3, velocity=1.0):
        mpc = do_mpc.controller.MPC(self.mpc_model)
        n_horizon = min(horizon, len(self.questions) - current_question_idx)
        if n_horizon <= 0:
            return 60

        suppress_ipopt = {
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'print_time': 0
        }
        mpc.set_param(
            n_horizon=n_horizon,
            t_step=1,
            store_full_solution=True,
            n_robust=1,
            nlpsol_opts=suppress_ipopt
        )
        time_alloc = self.mpc_model.u['time_alloc']
        current_question = self.questions[current_question_idx]
        # Use the *live* speaking velocity (words/sec) to affect predicted time
        predicted_duration = self.predict_duration_simple(current_question, velocity_factor=velocity)
        lterm = (time_alloc - ca.SX([predicted_duration])) ** 2
        mterm = ca.SX(0)
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.bounds['lower', '_u', 'time_alloc'] = np.array([[30]])
        mpc.bounds['upper', '_u', 'time_alloc'] = np.array([[300]])
        mpc.setup()
        x0 = np.array([remaining_time, current_question_idx]).flatten()
        mpc.x0 = x0
        mpc.set_initial_guess()
        with HiddenPrints():
            try:
                u0 = mpc.make_step(x0)
                return float(u0.item())
            except Exception as e:
                return float(min(remaining_time, 120))

    def predict_duration_simple(self, question, velocity_factor=1.0):
        DIFFICULTY_FACTORS = {
            "easy": 0.9,
            "medium": 1.0,
            "hard": 1.2
        }
        base = question.base_time
        difficulty = DIFFICULTY_FACTORS.get(question.difficulty, 1.0)
        category = self.compute_category_factor(question.category)
        return base * difficulty * velocity_factor * category

    def compute_category_factor(self, category):
        DEFAULT_CATEGORY_FACTOR = 1.0
        SMOOTHING_ALPHA = 0.7
        history = self.category_history.get(category, [])
        if not history:
            return DEFAULT_CATEGORY_FACTOR
        avg_actual = sum(h['actual'] for h in history) / len(history)
        avg_base = sum(h['base'] for h in history) / len(history)
        factor = avg_actual / avg_base if avg_base > 0 else DEFAULT_CATEGORY_FACTOR
        prev = self.prev_category_factors.get(category, DEFAULT_CATEGORY_FACTOR)
        smoothed = SMOOTHING_ALPHA * factor + (1 - SMOOTHING_ALPHA) * prev
        self.prev_category_factors[category] = smoothed
        return smoothed

    def update_category_history(self, question, allocated, actual):
        self.category_history[question.category].append({
            'allocated': allocated,
            'actual': actual,
            'base': question.base_time
        })

    def print_status(self):
        print("\nCategory Learning Status:")
        for category, history in self.category_history.items():
            if history:
                factor = self.prev_category_factors.get(category, 1.0)
                print(f"  {category}: {len(history)} questions, factor = {factor:.2f}")
            else:
                print(f"  {category}: no data yet")
