import os
import asyncio
import sqlite3
import time
import logging
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

# ---- Pipecat audio and pipeline imports ----
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.azure.tts import AzureTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.observers.base_observer import BaseObserver
from pipecat.frames.frames import (
    StartFrame, TTSSpeakFrame, EndFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
    TranscriptionFrame
)

# ---- Custom MPC controller ----
from MPC_module import MPCInterviewController, QuestionConfig

logging.getLogger('pipecat.services.whisper.base_stt').setLevel(logging.WARNING)
load_dotenv()

# ---- Global configuration ----
DB_PATH = os.path.join(os.getcwd(), "questions.db")
DEFAULT_SESSION_DURATION = 200  # Test duration (change back to 2700 for 45 min)
INACTIVITY_THRESHOLD = 20
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AZURE_TTS_KEY = os.getenv("AZURE_SPEECH_API_KEY")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION")


def fetch_questions_from_db() -> list:
    """Fetches questions from the database as QuestionConfig objects."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, question, difficulty, time, category FROM questions")
        rows = cursor.fetchall()
        conn.close()
        return [
            QuestionConfig(
                question_id=row[0],
                question_text=row[1],
                difficulty=row[2],
                base_time=float(row[3]),
                category=row[4]
            ) for row in rows
        ]
    except Exception as e:
        logger.error(f"❌ Failed to load questions: {e}")
        return []


class IntegratedInterviewManager:
    def __init__(self):
        self.questions_data = fetch_questions_from_db()
        self.answers_log = []
        self.task: Optional[PipelineTask] = None
        self.current_question_idx: int = 0
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.interview_complete = False
        self.session_ended = False
        self.session_start_time = None
        self.inactivity_task: Optional[asyncio.Task] = None
        self.answer_start_time = None
        self.answer_end_time = None
        self.transcription_buffer = []
        self.active_question_id = None
        self.is_answering = False
        self.waiting_for_final = False
        self.latest_transcript = ""
        self.prev_velocity = 0.0
        self.mpc_controller = MPCInterviewController(
            self.questions_data,
            total_session_time=DEFAULT_SESSION_DURATION
        )
        self.remaining_session_time = DEFAULT_SESSION_DURATION

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self.event_loop = loop

    def start_session(self):
        """Initialize session with all required state."""
        self.current_question_idx = 0
        self.remaining_session_time = DEFAULT_SESSION_DURATION
        self.session_ended = False
        self.session_start_time = time.time()
        logger.info(f"✅ Session started at {time.strftime('%H:%M:%S', time.localtime(self.session_start_time))}")

    def get_current_question(self) -> Optional[QuestionConfig]:
        if 0 <= self.current_question_idx < len(self.questions_data):
            return self.questions_data[self.current_question_idx]
        return None

    def get_elapsed_time(self) -> float:
        """Returns elapsed time since session start."""
        if self.session_start_time:
            return time.time() - self.session_start_time
        return 0.0

    def get_actual_remaining_time(self) -> float:
        """Returns actual remaining time based on start time."""
        elapsed = self.get_elapsed_time()
        return max(0, DEFAULT_SESSION_DURATION - elapsed)

    def compute_allocation(self) -> float:
        curidx = self.current_question_idx
        alloc = self.mpc_controller.optimize_allocation(
            current_question_idx=curidx,
            remaining_time=self.remaining_session_time,
            horizon=3
        )
        return min(alloc, self.remaining_session_time)

    def start_question_timing(self):
        """Start timing for a new question."""
        if self.session_ended:
            logger.warning("⚠️ Cannot start question - session has ended")
            return
        
        question = self.get_current_question()
        if not question or not self.event_loop:
            logger.error("❌ Missing question or event loop")
            return
        
        self.active_question_id = question.question_id
        self.transcription_buffer = []
        self.latest_transcript = ""
        self.is_answering = True
        self.answer_start_time = None
        self.answer_end_time = None
        self.waiting_for_final = False
        self.allocated_time = self.compute_allocation()
        self.cancel_inactivity_timer()
        self.start_inactivity_timer(self.current_question_idx)

    def cancel_inactivity_timer(self):
        if self.inactivity_task is not None and not self.inactivity_task.done():
            self.inactivity_task.cancel()
            self.inactivity_task = None

    def start_inactivity_timer(self, question_idx):
        loop = asyncio.get_event_loop()
        self.inactivity_task = loop.create_task(self.check_inactivity(question_idx))

    async def check_inactivity(self, question_idx):
        try:
            await asyncio.sleep(INACTIVITY_THRESHOLD)
            if self.is_answering and self.current_question_idx == question_idx and not self.answer_start_time:
                logger.warning(f"⚠️ No response for {INACTIVITY_THRESHOLD}s - auto-skipping")
                asyncio.create_task(self._handle_question_complete("[No response - skipped]", 0))
        except asyncio.CancelledError:
            logger.debug("🔄 Inactivity timer cancelled due to speech event.")

    def on_user_started_speaking(self):
        if not self.answer_start_time:
            self.answer_start_time = time.time()
        self.cancel_inactivity_timer()

    def on_user_stopped_speaking(self):
        if self.is_answering and self.answer_start_time:
            self.answer_end_time = time.time()
            self.waiting_for_final = True

    def append_to_answer(self, question_id: int, text: str):
        if text:
            self.latest_transcript = text
        if self.is_answering and self.active_question_id == question_id:
            self.transcription_buffer.append(text)
            logger.debug(f"✅ Transcription added to Q{question_id} buffer")

    async def _handle_question_complete(self, forced_answer, actual_time=None, transcript=None, velocity=None, words=None):
        """Handle completion of a question answer."""
        if self.session_ended:
            logger.info("⏰ Session ended during answer - finalizing")
            await self._end_interview(force_timeout=True)
            return
        
        if not self.is_answering:
            return
        
        self.is_answering = False
        self.active_question_id = None
        self.cancel_inactivity_timer()
        question = self.get_current_question()
        answer_text = forced_answer if forced_answer else (transcript or " ".join(self.transcription_buffer).strip())

        if actual_time is None and self.answer_start_time and self.answer_end_time:
            actual_time = self.answer_end_time - self.answer_start_time

        if actual_time is None or actual_time <= 0 or forced_answer:
            actual_time = 0.0

        num_words = words if words is not None else (len(answer_text.split()) if not forced_answer else 0)

        velo = velocity if velocity is not None else (num_words / actual_time if actual_time > 0 else 0.0)
        self.prev_velocity = velo
        allocated = self.allocated_time

        if question:
            self.mpc_controller.update_category_history(question, allocated=allocated, actual=actual_time)

        self.answers_log.append({
            "question_id": question.question_id if question else None,
            "question": question.question_text if question else "",
            "category": question.category if question else "",
            "difficulty": question.difficulty if question else "",
            "answer": answer_text,
            "allocated_time": allocated,
            "actual_time": actual_time,
            "velocity": velo,
            "words": num_words,
            "time_saved": allocated - actual_time
        })

        logger.info(
            f"💾 Q{question.question_id if question else '?'}: "
            f"Allocated: {allocated:.1f}s | Actual: {actual_time:.1f}s | "
            f"Velocity: {velo:.2f} words/sec | "
            f"Status: {'✅ On time' if allocated - actual_time >= 0 else '⚠️ Exceeded'}"
        )

        self.remaining_session_time -= actual_time
        
        if self.remaining_session_time <= 0:
            logger.warning("⏰ Session time exhausted after this question")
            self.session_ended = True
        
        self.current_question_idx += 1
        await asyncio.sleep(1)
        await self._proceed_to_next()

    async def _proceed_to_next(self):
        """Proceed to next question or end interview."""
        if self.session_ended or self.remaining_session_time <= 0:
            logger.warning("⏰ Session time exhausted - ending interview")
            await self._end_interview(force_timeout=True)
            return
        
        if self.current_question_idx < len(self.questions_data):
            next_q = self.get_current_question()
            allocation = self.compute_allocation()
            
            if allocation > self.remaining_session_time:
                logger.warning(
                    f"⚠️ Insufficient time for next question "
                    f"(need {allocation:.1f}s, have {self.remaining_session_time:.1f}s)"
                )
                await self._end_interview(force_timeout=False)
                return
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📌 Next Question (Q{next_q.question_id})")
            logger.info(f"   Question: {next_q.question_text}")
            logger.info(f"   Category: {next_q.category} | Difficulty: {next_q.difficulty}")
            logger.info(f"   MPC Allocated: {int(allocation)}s | Remaining: {int(self.remaining_session_time)}s")
            logger.info(f"{'='*60}\n")
            self.start_question_timing()
            
            if self.task:
                await self.task.queue_frames([
                    TTSSpeakFrame(next_q.question_text)
                ])
        else:
            await self._end_interview()

    async def start_first_question(self):
        """Queue the first question to start the interview."""
        if not self.questions_data:
            logger.error("❌ No questions available to start interview")
            return
        
        first_q = self.get_current_question()
        allocation = self.compute_allocation()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📌 Starting Interview with Question (Q{first_q.question_id})")
        logger.info(f"   Question: {first_q.question_text}")
        logger.info(f"   Category: {first_q.category} | Difficulty: {first_q.difficulty}")
        logger.info(f"   MPC Allocated: {int(allocation)}s | Remaining: {int(self.remaining_session_time)}s")
        logger.info(f"{'='*60}\n")
        
        self.start_question_timing()
        
        if self.task:
            # Queue greeting and first question
            await self.task.queue_frames([
                TTSSpeakFrame("Hello! Welcome to your interview session. Let's begin with the first question."),
                TTSSpeakFrame(first_q.question_text)
            ])

    async def _end_interview(self, force_timeout=False):
        """End the interview session."""
        if self.interview_complete:
            return
        
        self.interview_complete = True
        self.cancel_inactivity_timer()
        
        logger.info("\n============================================")
        logger.info("✅ Interview Session Complete")
        logger.info("============================================")
        
        elapsed_time = self.get_elapsed_time()
        logger.info(f"\n⏱️ Session Duration: {int(elapsed_time)}s / {DEFAULT_SESSION_DURATION}s")
        logger.info(f"📊 Interview Summary:\n  Total questions answered: {len(self.answers_log)}")
        
        for ans in self.answers_log:
            logger.info(
                f"\n  Q{ans['question_id']} ({ans['category']}, {ans['difficulty']}):"
                f"\n      Question: {ans['question'][:60]}..."
                f"\n      Time: {ans['actual_time']:.1f}s / {ans['allocated_time']:.1f}s |"
                f" Velocity: {ans['velocity']:.2f} words/sec | Words: {ans['words']}"
            )
            if not ans['answer'].startswith("["):
                logger.info(f"      Answer: {ans['answer'][:80]}...")
        
        self.mpc_controller.print_status()
        
        if self.task:
            closing_message = (
                "Thank you for completing the interview." +
                (" The session time limit has been reached." if force_timeout else " Goodbye!")
            )
            await self.task.queue_frames([
                TTSSpeakFrame(closing_message),
                EndFrame()
            ])
            await asyncio.sleep(3)

    async def enforce_session_timeout(self):
        """Enforce the session timeout."""
        await asyncio.sleep(DEFAULT_SESSION_DURATION)
        if not self.interview_complete:
            logger.warning(f"⏰ {DEFAULT_SESSION_DURATION}s session limit reached - ending interview")
            self.session_ended = True
            self.is_answering = False
            self.cancel_inactivity_timer()
            await self._end_interview(force_timeout=True)


class InterviewSpeechObserver(BaseObserver):
    """Observer to handle speech events from the pipeline."""
    
    async def on_push_frame(self, data):
        frame = getattr(data, "frame", None)
        if frame is None or isinstance(frame, StartFrame):
            return
        
        if isinstance(frame, UserStartedSpeakingFrame):
            manager.on_user_started_speaking()
            logger.debug("🎤 Detected user started speaking")
        
        elif isinstance(frame, UserStoppedSpeakingFrame):
            manager.on_user_stopped_speaking()
            logger.debug("🎤 Detected user stopped speaking")
        
        elif isinstance(frame, TranscriptionFrame):
            text = getattr(frame, "text", "")
            manager.append_to_answer(manager.active_question_id, text)
            
            if manager.is_answering and manager.waiting_for_final:
                manager.waiting_for_final = False
                transcript = text.strip()
                num_words = len(transcript.split())
                actual_time = (
                    manager.answer_end_time - manager.answer_start_time 
                    if (manager.answer_start_time and manager.answer_end_time) 
                    else 0
                )
                velocity = num_words / actual_time if actual_time > 0 else 0.0
                
                if transcript:
                    logger.debug(f"🗣️ Full transcription: [{transcript}] | Words: {num_words} | Velo: {velocity:.2f}")
                
                asyncio.run_coroutine_threadsafe(
                    manager._handle_question_complete(None, actual_time, transcript, velocity, num_words),
                    manager.event_loop
                )


manager = IntegratedInterviewManager()


async def main():
    """Main entry point for the interview system."""
    logger.info("🚀 Starting MPC-Pipecat Interview System")
    
    if not GROQ_API_KEY:
        logger.error("❌ GROQ_API_KEY must be set")
        return
    if not AZURE_TTS_KEY or not AZURE_REGION:
        logger.error("❌ Missing Azure Speech credentials")
        return
    if not manager.questions_data:
        logger.error("❌ No questions found in database.")
        return
    
    loop = asyncio.get_event_loop()
    manager.set_event_loop(loop)
    
    vad_params = VADParams(stop_secs=8, start_secs=0.3)
    vad = SileroVADAnalyzer(params=vad_params)
    
    stt = GroqSTTService(api_key=GROQ_API_KEY, model="whisper-large-v3-turbo")
    tts = AzureTTSService(
        api_key=AZURE_TTS_KEY, 
        region=AZURE_REGION, 
        voice="en-US-JennyNeural", 
        sample_rate=24000
    )
    
    transport = LocalAudioTransport(params=LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=vad
    ))
    
    pipeline = Pipeline([
        transport.input(),
        stt,
        tts,
        transport.output()
    ])
    
    observer = InterviewSpeechObserver()
    task = PipelineTask(pipeline, params=PipelineParams(
        allow_interruptions=False,
        enable_metrics=True,
        observers=[observer]
    ))
    
    manager.task = task
    manager.start_session()
    
    # Start the first question BEFORE running the pipeline
    await manager.start_first_question()
    
    timeout_task = asyncio.create_task(manager.enforce_session_timeout())
    runner = PipelineRunner()
    runner_task = asyncio.create_task(runner.run(task))
    
    try:
        done, pending = await asyncio.wait(
            [timeout_task, runner_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task_obj in pending:
            logger.info(f"⏰ Cancelling pending task")
            task_obj.cancel()
            try:
                await task_obj
            except asyncio.CancelledError:
                logger.debug("✅ Task cancelled successfully")
            except Exception as e:
                logger.error(f"❌ Error during task cancellation: {e}")
        
    except asyncio.CancelledError:
        logger.info("⏰ Session terminated by cancellation")
    except Exception as e:
        logger.error(f"❌ Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not manager.interview_complete:
            await manager._end_interview(force_timeout=True)
        logger.info("✅ Main function cleanup complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interview interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
