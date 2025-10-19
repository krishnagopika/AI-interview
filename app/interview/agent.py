from google.adk.agents import Agent, LlmAgent
from .tools.get_current_time import get_current_time
from .tools.question_selector import fetch_questions
from .tools.answer_evaluator import evaluate_answer
from dotenv import load_dotenv
import os

load_dotenv()




# ------------------- QUESTION SELECTION AGENT -------------------
question_selector_agent = LlmAgent(
    name="question_selector",
    model="gemini-2.0-flash-exp",
    description="Selects interview questions based on topic and difficulty.",
    instruction="""
    You are the Question Selector Agent.
    - Retrieve questions from the RAG tool based on topic, difficulty (easy, medium, hard), and format.
    - Provide only the question text and an internal ID.
    - Always consider candidate skill level and previous questions asked.
    - Avoid duplicates and trivial questions.
    """,
    tools=[fetch_questions],
)

# ------------------- EVALUATOR AGENT -------------------
evaluator_agent = LlmAgent(
    name="answer_evaluator",
    model="gemini-2.0-flash-exp",
    description="Evaluates candidate answers and provides scores and feedback.",
    instruction="""
    You are the Evaluator Agent.
    - Evaluate answers based on correctness, completeness, and clarity.
    - Provide a score from 0-10 and concise feedback.
    - Only return structured feedback: {{'score': <int>, 'feedback': <str>}}.
    - NEVER expose raw tool outputs.
    """,
    tools=[evaluate_answer],
)

# ------------------- INTERVIEW AGENT -------------------
interview_agent = LlmAgent(
    name="interview_agent",
    model="gemini-2.0-flash-exp",
    description="Conducts live interviews with candidates, asking questions and collecting answers.",
    instruction=f"""
    You are the Interview Agent.
    - Conduct a live conversation with the candidate.
    - Ask questions retrieved from the Question Selector Agent.
    - After each candidate answer, send it to the Evaluator Agent to get a score and feedback.
    - Provide concise feedback to the candidate only after evaluation.
    - Keep conversation interactive and professional.
    - Track which questions have been asked to avoid repeats.
    Today's date is {get_current_time()}.
    """,
    tools=[fetch_questions, evaluate_answer],  # uses both RAG fetch and evaluation tools
)

coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.0-flash-exp",
    description="I coordinate greetings and tasks.",
    sub_agents=[ # Assign sub_agents here
        interview_agent,
        evaluator_agent,
        question_selector_agent

    ]
)


# ------------------- MULTI-AGENT FLOW -------------------
# Example pseudo-flow:
# 1. Interview agent asks question
# 2. Question selector chooses question from RAG
# 3. Candidate answers
# 4. Evaluator scores answer
# 5. Interview agent continues to next question
