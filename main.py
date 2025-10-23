"""
Main execution script for the MPC-integrated interview system.
"""
import traceback
from config import QUESTIONS, DEFAULT_SESSION_DURATION
from time_controller import TimeController
from groq_engine import GROQEngine
from interview_manager import InterviewManager


def main():
    """
    Main entry point for the MPC-integrated interview system.
    
    This runs a complete interview with:
    - Configurable questions across different categories
    - Real-time MPC-based time allocation
    - Text input from candidate
    - Performance tracking and learning
    - Auto-skip after inactivity threshold
    """
    
    print("\n" + "="*60)
    print("MPC-POWERED INTERVIEW SYSTEM")
    print("="*60)
    print("\nFeatures:")
    print("• Model Predictive Control for adaptive time allocation")
    print("• Real-time learning from performance data")
    print("• Per-question timeout handling")
    print("• Session-level time management")
    print("• Auto-skip after inactivity threshold")
    print("="*60)
    
    # Initialize components
    gpt_engine = GROQEngine()
    time_controller = TimeController(
        questions=QUESTIONS,
        session_duration=DEFAULT_SESSION_DURATION
    )
    manager = InterviewManager(gpt_engine, time_controller)
    
    # Run the interview
    try:
        manager.run_interview()
    except KeyboardInterrupt:
        print("\n\n Interview interrupted by user.")
        print("Saving progress...")
        manager.end_interview()
    except Exception as e:
        print(f"\n\n Error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

