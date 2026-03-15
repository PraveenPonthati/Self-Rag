"""
LangGraph state definition.

All nodes in the Self-RAG graph share this typed state dictionary.
LangGraph passes it between nodes and merges updates automatically.
"""

from typing import List, Optional
from typing_extensions import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the full state carried through the LangGraph pipeline.

    Fields
    ------
    question : str
        The user's original (or rewritten) query.
    original_question : str
        The very first question the user asked — preserved across rewrites.
    documents : List[Document]
        Chunks retrieved from ChromaDB for the current question.
    relevant_documents : List[Document]
        Subset of `documents` that passed the relevance grader.
    generation : str
        The LLM's answer, populated after the generate node.
    hallucination_score : str
        "yes" if the answer is grounded, "no" if it is not.
    answer_grade : str
        "yes" if the answer actually addresses the question, "no" otherwise.
    retry_count : int
        How many query-rewrite attempts have been made so far.
    error : Optional[str]
        Error message if something went wrong (shown in the UI).
    """

    question: str
    original_question: str
    documents: List[Document]
    relevant_documents: List[Document]
    generation: str
    hallucination_score: str
    answer_grade: str
    retry_count: int
    error: Optional[str]
