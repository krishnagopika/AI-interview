import os
import random
from uuid import uuid4
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ---------------- ENV + SETUP ----------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "tech-questions")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    print(f"Creating index '{INDEX_NAME}' ...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # embedding size for all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# ---------------- FUNCTION ----------------
def fetch_questions(topic: str = "", difficulty: str = "Medium", previous_ids=None, max_results: int = 1):
    """
    Fetch questions from Pinecone based on topic, difficulty, and previously asked question IDs.

    Args:
        topic (str): Topic or skill area (e.g., 'Spring Boot', 'React').
        difficulty (str): 'Easy', 'Medium', or 'Hard'.
        previous_ids (list): List of question IDs already asked.
        max_results (int): Number of questions to return.

    Returns:
        list[dict]: [{'id', 'question', 'answer', 'difficulty', 'skill', 'unit'}]
    """
    previous_ids = previous_ids or []

    # --- Step 1: Create embedding for topic ---
    query_text = topic if topic else "technical interview question"
    query_embedding = model.encode(query_text).tolist()

    # --- Step 2: Query Pinecone ---
    response = index.query(
        vector=query_embedding,
        top_k=20,
        include_metadata=True
    )

    # --- Step 3: Filter out previous IDs ---
    results = []
    for match in response.matches:
        meta = match.metadata
        if match.id not in previous_ids:
            results.append({
                "id": match.id,
                "question": meta.get("question"),
                "answer": meta.get("answer"),
                "difficulty": meta.get("difficulty"),
                "skill": meta.get("skill"),
                "unit": meta.get("unit"),
                "score": match.score
            })

    # --- Step 4: Randomize and return subset ---
    random.shuffle(results)
    return results[:max_results]


# ---------------- TEST / DEMO ----------------
if __name__ == "__main__":
    questions = fetch_questions(topic="Spring Boot", difficulty="Medium")
    for q in questions:
        print(f"\nID: {q['id']}")
        print(f"Question: {q['question']}")
        print(f"Answer: {q['answer']}")
        print(f"Skill: {q['skill']} | Unit: {q['unit']} | Difficulty: {q['difficulty']}")
        print("-" * 100)
