"""
LangGraph node implementations.

Each function takes the current GraphState and returns a dict of updates.
LangGraph merges those updates back into the state automatically.

Nodes in the graph:
    retrieve            → fetch top-k chunks from ChromaDB
    grade_documents     → filter out irrelevant chunks using the LLM
    generate            → produce an answer from the relevant chunks
    check_hallucination → verify the answer is grounded in the documents
    rewrite_query       → rewrite the question and reset retrieval state
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GOOGLE_API_KEY, LLM_MODEL, MAX_RETRIES
from src.graph_state import GraphState
from src.vector_store import retrieve_documents

# ── Shared LLM instance (Gemini) ─────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0,           # deterministic outputs for grading/generation
    google_api_key=GOOGLE_API_KEY,
)

# ── Prompts ───────────────────────────────────────────────────────────────────

RELEVANCE_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance grader. "
            "Given a retrieved document chunk and a user question, "
            "decide whether the chunk contains information that is useful "
            "for answering the question.\n"
            "Respond with a single word: 'yes' if relevant, 'no' if not.",
        ),
        (
            "human",
            "Question: {question}\n\nDocument chunk:\n{document}",
        ),
    ]
)

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions strictly based on "
            "the provided context documents. "
            "If the context does not contain enough information to answer, "
            "say so explicitly. Do not make up facts.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a hallucination detector. "
            "Given a context and an answer, determine whether every claim in "
            "the answer is supported by the context.\n"
            "Respond with 'yes' if the answer is fully grounded, "
            "'no' if it contains unsupported claims.",
        ),
        (
            "human",
            "Context:\n{context}\n\nAnswer:\n{answer}",
        ),
    ]
)

ANSWER_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an answer quality evaluator. "
            "Given a question and an answer, determine whether the answer "
            "meaningfully addresses the question.\n"
            "Respond with 'yes' if it does, 'no' if it does not.",
        ),
        (
            "human",
            "Question: {question}\n\nAnswer: {answer}",
        ),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a search query optimizer. "
            "The initial query failed to retrieve useful documents. "
            "Rewrite the query to be more specific, use different vocabulary, "
            "and increase the chance of finding relevant information.\n"
            "Return only the rewritten query — no explanation.",
        ),
        (
            "human",
            "Original question: {question}",
        ),
    ]
)

# ── Chain builders ────────────────────────────────────────────────────────────
_parser = StrOutputParser()

relevance_chain = RELEVANCE_GRADER_PROMPT | llm | _parser
generation_chain = GENERATION_PROMPT | llm | _parser
hallucination_chain = HALLUCINATION_PROMPT | llm | _parser
answer_grade_chain = ANSWER_GRADE_PROMPT | llm | _parser
rewrite_chain = REWRITE_PROMPT | llm | _parser


# ── Node functions ────────────────────────────────────────────────────────────

def retrieve(state: GraphState) -> dict:
    """
    Node: retrieve
    Fetches the top-k most relevant document chunks from ChromaDB
    for the current question stored in state.
    """
    question = state["question"]
    documents = retrieve_documents(question)
    return {"documents": documents, "relevant_documents": []}


def grade_documents(state: GraphState) -> dict:
    """
    Node: grade_documents
    Iterates over each retrieved chunk and asks the LLM whether it is
    relevant to the question. Keeps only the chunks that pass.
    """
    question = state["question"]
    documents = state["documents"]

    relevant: List[Document] = []
    for doc in documents:
        score = relevance_chain.invoke(
            {"question": question, "document": doc.page_content}
        )
        if score.strip().lower().startswith("yes"):
            relevant.append(doc)

    return {"relevant_documents": relevant}


def generate(state: GraphState) -> dict:
    """
    Node: generate
    Concatenates the relevant chunks into a single context string and
    prompts Gemini to produce an answer.
    """
    question = state["question"]
    relevant_docs = state["relevant_documents"]

    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in relevant_docs
    )

    generation = generation_chain.invoke(
        {"context": context, "question": question}
    )

    return {"generation": generation}


def check_hallucination(state: GraphState) -> dict:
    """
    Node: check_hallucination
    Verifies the answer is grounded in the context and actually addresses
    the question. Sets hallucination_score and answer_grade accordingly.
    """
    generation = state["generation"]
    relevant_docs = state["relevant_documents"]

    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

    hallucination_score = hallucination_chain.invoke(
        {"context": context, "answer": generation}
    )

    answer_grade = answer_grade_chain.invoke(
        {"question": state["original_question"], "answer": generation}
    )

    return {
        "hallucination_score": hallucination_score.strip().lower()[:3],
        "answer_grade": answer_grade.strip().lower()[:3],
    }


def rewrite_query(state: GraphState) -> dict:
    """
    Node: rewrite_query
    Rewrites the user's question and increments the retry counter so the
    graph loops back to retrieve with a better query.
    """
    question = state["original_question"]
    retry_count = state.get("retry_count", 0)

    new_question = rewrite_chain.invoke({"question": question})

    return {
        "question": new_question.strip(),
        "retry_count": retry_count + 1,
        "documents": [],
        "relevant_documents": [],
        "generation": "",
    }
