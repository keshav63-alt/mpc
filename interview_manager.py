class InterviewManager:
    """
    Main interview manager that orchestrates the interview process.
    
    Features:
    - Integrates MPC-based time control
    - Manages question flow
    - Handles timeouts gracefully
    - Tracks performance metrics
    - Provides real-time feedback
    - Auto-skips questions after inactivity threshold
    """
    
    def __init__(self, groq_engine, time_controller):
        """
        Initialize the interview manager.
        
        Args:
            groq_engine: GPT engine for question generation and evaluation
            time_controller: Time controller with MPC integration
        """
        self.gpt = groq_engine
        self.timer = time_controller
        self.answers = []  # Store all answers for later analysis

    def run_interview(self):
        """
        Main interview loop that processes all questions.
        
        Workflow for each question:
        1. Check if time/questions remain
        2. Get MPC-optimized time allocation
        3. Present question to candidate
        4. Start question timer and inactivity monitor
        5. Collect candidate response
        6. Stop timers and record actual time
        7. Evaluate response
        8. Update MPC with performance data
        9. Move to next question
        """
        # Start the session timer
        self.timer.start_session()
        
        # Main interview loop
        while not self.timer.is_time_up() and self.timer.has_more_questions():
            
            # Get current question
            question = self.timer.get_current_question()
            
            # Get MPC-optimized time allocation
            allocated_time = self.timer.get_mpc_allocation()
            
            # Display allocation info
            print(f"\nTime Allocation (MPC-optimized)")
            print(f"   Allocated: {allocated_time:.1f} seconds ({allocated_time/60:.1f} min)")
            print(f"   Remaining Session Time: {self.timer.get_remaining_time():.1f}s")
            print(f"   Auto-skip after {self.timer.INACTIVITY_THRESHOLD}s of no input")
            
            # Present question to candidate
            self.gpt.present_question(question)
            
            # Start question timer with timeout callback
            self.timer.start_question_timer(
                limit_seconds=allocated_time,
                on_timeout_callback=self.handle_timeout
            )
            
            # Start inactivity monitor
            self.timer.start_inactivity_monitor(
                on_inactivity_callback=self.handle_inactivity
            )
            
            # Get candidate response (with timeout and inactivity handling)
            candidate_answer = self.get_candidate_response()
            
            # Stop timers and get actual time spent
            actual_time = self.timer.stop_question_timer()
            
            # Store answer for later analysis
            self.answers.append({
                'question': question,
                'answer': candidate_answer,
                'allocated_time': allocated_time,
                'actual_time': actual_time
            })
            
            # Display timing results
            time_diff = allocated_time - actual_time
            if time_diff > 0:
                print(f"\nTiming: Finished in {actual_time:.1f}s (saved {time_diff:.1f}s)")
            else:
                print(f"\n Timing: Finished in {actual_time:.1f}s (exceeded by {abs(time_diff):.1f}s)")
            
            # Evaluate answer with GPT
            self.gpt.evaluate_answer(candidate_answer, question)
            
            # Update MPC controller with performance data for learning
            self.timer.update_mpc_with_performance(
                question=question,
                allocated_time=allocated_time,
                actual_time=actual_time
            )
            
            # Move to next question
            self.timer.move_to_next_question()
        
        # End of interview
        self.end_interview()

    def get_candidate_response(self):
        """
        Capture candidate's text input response with inactivity detection.
        
        This method handles:
        - Multi-line input (type 'DONE' on new line to submit)
        - Timeout handling (if timer expires during input)
        - Inactivity handling (if no input for threshold seconds)
        - Empty response handling
        
        Returns:
            str: Candidate's answer text
        """
        print("Your Answer (type your response, then type 'DONE' on a new line to submit):")
        print("   (or press Ctrl+C to skip)")
        print(f"   Will auto-skip after {self.timer.INACTIVITY_THRESHOLD} seconds of no input\n")
        
        lines = []
        try:
            while True:
                # Wait for user input
                line = input()
                
                # Record activity when user provides input
                self.timer.record_activity()
                
                # Check for submission command
                if line.strip().upper() == 'DONE':
                    break
                
                # Check if timeout occurred during input
                if self.timer.timeout_occurred.is_set():
                    print("\n Time expired while typing!")
                    break
                
                # Check if inactivity occurred
                if self.timer.inactivity_occurred.is_set():
                    print("\n Moving to next question due to inactivity...")
                    break
                
                lines.append(line)
        
        except KeyboardInterrupt:
            print("\n Skipping question...")
            return "[No response provided]"
        
        # Join all lines into single answer
        answer = '\n'.join(lines).strip()
        
        # Handle empty response
        if not answer:
            answer = "[No response provided]"
        
        return answer

    def handle_timeout(self):
        """
        Handle timeout event during question answering.
        
        Called by the timer thread when allocated time expires.
        """
        self.gpt.handle_timeout()

    def handle_inactivity(self):
        """
        Handle inactivity event during question answering.
        
        Called by the inactivity monitor when user is inactive for threshold seconds.
        """
        self.gpt.handle_inactivity()

    def end_interview(self):
        """
        Finalize the interview and display summary statistics.
        
        Shows:
        - Completion status
        - Questions answered
        - Time usage
        - MPC learning summary
        """
        # Calculate session duration
        session_time = int(self.timer.session_duration - self.timer.get_remaining_time())
        
        # Check completion status
        if self.timer.is_time_up():
            print(f"\n Session time exhausted!")
        elif not self.timer.has_more_questions():
            print(f"\n All questions completed!")
        
        # Generate closing remarks
        self.gpt.generate_closing(
            total_questions=len(self.answers),
            session_time=session_time
        )
        
        # Display MPC learning summary
        self.timer.print_mpc_learning_status()
        
        # Display per-question summary
        self.print_session_summary()

    def print_session_summary(self):
        """Display detailed summary of all questions and timing."""
        print(f"{'='*60}")
        print("Session Summary")
        print(f"{'='*60}")
        
        for i, item in enumerate(self.answers, 1):
            q = item['question']
            allocated = item['allocated_time']
            actual = item['actual_time']
            diff = allocated - actual
            
            status = "Saved time" if diff > 0 else "  Used extra time"
            
            print(f"\nQ{i}: {q.question_text[:40]}...")
            print(f"   Category: {q.category} | Difficulty: {q.difficulty}")
            print(f"   Allocated: {allocated:.1f}s | Actual: {actual:.1f}s | {status}: {abs(diff):.1f}s")
        
        print(f"\n{'='*60}\n")
