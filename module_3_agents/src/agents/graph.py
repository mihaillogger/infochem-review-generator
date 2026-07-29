from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import planner_node, external_writer_node, compiler_node


def route_sections(state: GraphState) -> str:
    """Определяет, есть ли еще секции для написания."""
    if state["current_section"] is not None:
        return "external_writer_node"
    return "compiler_node"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("planner_node", planner_node)

    # Единый узел Матвея (Adapter + Writer)
    workflow.add_node("external_writer_node", external_writer_node)

    workflow.add_node("compiler_node", compiler_node)

    workflow.set_entry_point("planner_node")
    workflow.add_edge("planner_node", "external_writer_node")

    workflow.add_conditional_edges(
        "external_writer_node",
        route_sections,
        {
            "external_writer_node": "external_writer_node",
            "compiler_node": "compiler_node"
        }
    )

    workflow.add_edge("compiler_node", END)
    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()

    initial_state = {
        "global_topic": "Advancements in Catalytic Conversion of Biomass",
        "pending_sections": [],
        "current_section": None,
        "completed_sections": [],
        "final_document": ""
    }

    result = app.invoke(initial_state)
    print(result["final_document"])