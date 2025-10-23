MPC Interview AI System
A modular, voice-based interview platform leveraging Model Predictive Control (MPC) for adaptive time allocation. This solution enables highly dynamic, efficient, and fair interview sessions powered by advanced Python, AI, and speech technologies.





🚀 Features
MPC-Based Question Timing: Optimal, adaptive allocation of time per question using Model Predictive Control techniques.

Voice-Driven Interviews: Seamless integration with Groq/OpenAI for real-time Speech-to-Text and Text-to-Speech.

Session & Inactivity Management: Session timers, automatic question skip after user inactivity, and complete session summaries.

Extensible Architecture: Clean modular files — easily customize, extend, or integrate with other AI services.



File                  |  Purpose                                                
----------------------+---------------------------------------------------------
MPC_module.py         |  Core MPC logic for dynamic time control                
time_controller.py    |  Timing logic, inactivity/skipping, and session handling
main.py               |  Application entry point                                
config.py             |  Configuration & constants                              
groq_engine.py        |  Groq/OpenAI voice engine integration                   
interview_manager.py  |  High-level interview flow manager                      
requirement.txt       |  Python dependencies                   


⚡ Getting Started

1. Clone the repo
  git clone https://github.com/keshav63-alt/mpc.git
  cd mpc

2. Install dependencies
  pip install -r requirement.txt


3. Set up environment variables
  GROQ_API_KEY=your_groq_key

4. Run the application
  python main.py

