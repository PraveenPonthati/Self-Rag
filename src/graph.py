"""
LangGraph workflow definition.

This module assembles the Self-RAG state machine by:
  1. Registering all nodes
  2. Defining edges (fixed and conditional)
  3. Compiling the graph into a runnable

Graph topology
──────────────
            ┌─────────────────────────────────┐
            │              START               │
            └──────────────┬──────────────────┘
                           │
                           ▼
                        retrieve
                           │
                           ▼
                    grade_documents
                           │
              ┌────────────┴────────────┐
              │ no relevant docs        │ relevant docs found
              ▼                         ▼
        rewrite_query               generate
              │                         │
              │ (retry < MAX_RETRIES)    ▼
              └──────────►   check_hallucination
                                        │
                          ┌─────────────┴──────────────┐
                          │ grounded & good answer      │ not grounded / bad answer
                          ▼                             ▼
                         END                      rewrite_query
                                                        │
                                                (retry exhausted → END)
"""

from langgraph.graph import END, StateGraph

from src.config import MAX_RETRIES
from src.graph_state import GraphState
from src.nodes import (
    check_hallucination,
    generate,
    grade_documents,
    retrieve,
    rewrite_query,
)


# ── Conditional edge functions ────────────────────────────────────────────────

def decide_after_grading(state: GraphState) -> str:
    """
    After grade_documents:
      - If there are relevant documents → go to generate
      - If none remain AND retries are not exhausted → rewrite the query
      - If retries exhausted → end (nothing useful found)
    """
    relevant = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)

    if relevant:
        return "generate"
    elif retry_count < MAX_RETRIES:
        return "rewrite_query"
    else:
        return END


def decide_after_hallucination_check(state: GraphState) -> str:
    """
    After check_hallucination:
      - If answer is grounded ('yes') AND addresses the question ('yes') → END
      - Otherwise, attempt a rewrite (if retries remain) or end anyway
    """
    hallucination = state.get("hallucination_score", "no")
    answer_grade = state.get("answer_grade", "no")
    retry_count = state.get("retry_count", 0)

    if hallucination == "yes" and answer_grade == "yes":
        return END
    elif retry_count < MAX_RETRIES:
        return "rewrite_query"
    else:
        # Retries exhausted — return whatever we have
        return END


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the Self-RAG LangGraph state machine.

    Returns:
        A compiled LangGraph app ready to invoke with an initial state dict.
    """
    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("rewrite_query", rewrite_query)

    # Entry point
    workflow.set_entry_point("retrieve")

    # Fixed edges
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("generate", "check_hallucination")

    # After rewrite, always go back to retrieve
    workflow.add_edge("rewrite_query", "retrieve")

    # Conditional edges
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "check_hallucination",
        decide_after_hallucination_check,
        {
            END: END,
            "rewrite_query": "rewrite_query",
        },
    )

    return workflow.compile()


# Singleton — import this in other modules instead of calling build_graph() each time
rag_graph = build_graph()
