# answer_evaluator.py

def evaluate_answer(question_id, candidate_answer):
    """
    Evaluate candidate answer and provide score and feedback.
    Arguments:
        question_id (str): The ID of the question answered
        candidate_answer (str): The candidate's answer
    Returns:
        dict: {'score': int (0-10), 'feedback': str}
    """
    answer_keypoints = {
        "q1": ["list", "ordered", "collection"],
        "q2": ["list comprehension", "syntax", "for loop", "expression"],
        "q3": ["GIL", "Global Interpreter Lock", "threads", "concurrency"],
        "q4": ["binary search", "sorted array", "log(n)"],
        "q5": ["Dijkstra", "graph", "shortest path", "priority queue"],
        "q6": ["A*", "heuristic", "graph", "priority queue"],
    }

    keypoints = answer_keypoints.get(question_id, [])
    score = sum(1 for kp in keypoints if kp.lower() in candidate_answer.lower())
    max_score = len(keypoints)
    normalized_score = int((score / max_score) * 10) if max_score > 0 else 0

    if normalized_score >= 8:
        feedback = "Excellent answer."
    elif normalized_score >= 5:
        feedback = "Good answer, but could be more detailed."
    else:
        feedback = "Answer is insufficient; consider reviewing key concepts."

    return {"score": normalized_score, "feedback": feedback}


