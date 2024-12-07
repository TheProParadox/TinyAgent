import networkx as nx

def construct_digraph_from_dict(task_dict: dict[int, dict[str]]) -> nx.DiGraph:
    task_graph=  nx.DiGraph()
    task_graph.add_nodes_from(task_dict.items())
    for cur_task_id, task_attributes in task_dict.items():
        for parent_task_id in task_attributes["dependencies"]:
            task_graph.add_edge(parent_task_id, cur_task_id)
    
    return task_graph


def are_task_nodes_equal(task_node_1_attributes: dict[str], task_node_2_attributes: dict[str]) -> bool:
    return task_node_1_attributes["tool_name"]==task_node_2_attributes["tool_name"]


def are_task_graphs_isomorphic(task_graph_1: nx.DiGraph, task_graph_2: nx.DiGraph) -> bool:
    return nx.is_isomorphic(task_graph_1, task_graph_2, node_match=are_task_nodes_equal)
