from langgraph.graph import StateGraph, END
import config
from state import GraphState
from nodes import (
    planner_node,
    adapter_node,
    eval_retriever_node,
    writer_node,
    eval_citation_node,
    advance_section_node,
    compiler_node,
)


def route_retriever(state: GraphState) -> str:
    curr = state["current_section"]
    if curr["retriever_rejection"] == "":
        return "writer_node"
    if curr["retriever_retries"] >= config.MAX_RETRIES:  # Лимит исчерпан
        return "writer_node"
    return "adapter_node"


def route_citation(state: GraphState) -> str:
    curr = state["current_section"]
    if curr["citation_errors"] == "":
        return "advance_section_node"
    if curr["writer_retries"] >= config.MAX_RETRIES:
        return "advance_section_node"
    return "writer_node"


def route_sections(state: GraphState) -> str:
    if state["current_section"] is not None:
        return "adapter_node"
    return "compiler_node"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("planner_node", planner_node)
    workflow.add_node("adapter_node", adapter_node)
    workflow.add_node("eval_retriever_node", eval_retriever_node)
    workflow.add_node("writer_node", writer_node)
    workflow.add_node("eval_citation_node", eval_citation_node)
    workflow.add_node("advance_section_node", advance_section_node)
    workflow.add_node("compiler_node", compiler_node)

    workflow.set_entry_point("planner_node")

    workflow.add_edge("planner_node", "adapter_node")
    workflow.add_edge("adapter_node", "eval_retriever_node")

    workflow.add_conditional_edges("eval_retriever_node", route_retriever)

    workflow.add_edge("writer_node", "eval_citation_node")
    workflow.add_conditional_edges("eval_citation_node", route_citation)

    workflow.add_conditional_edges(
        "advance_section_node",
        route_sections,
        {"adapter_node": "adapter_node", "compiler_node": "compiler_node"},
    )

    workflow.add_edge("compiler_node", END)
    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"global_topic": "Nanostructuring of metals"})
    print(result["final_document"])
