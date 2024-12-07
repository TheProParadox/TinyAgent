from src.llm_compiler.task_fetching_unit import Task
from src.llm_compiler.output_parser import _get_dependencies_from_graph

def convert_tasks_to_dicts(tasks: dict[int, Task]) -> dict[int, dict[str]]:
    task_dict = {
        task.idx: {
            "tool_name": task.name,
            "tool_args": task.args,
            "dependencies": task.dependencies
        }
        for task in tasks.values()
    }
    return task_dict


def flatten_list(l: list) -> list[str]:
    flat_list = []
    for element in l:
        if isinstance(element, list):
            flat_list.extend(flatten_list(element))
        else:
            flat_list.append(element)
    
    return flat_list


def get_dependencies(tool_idx: int, tool_name: str, tools_args: list) -> list[int]:
    flat_args_list = flatten_list(tools_args)
    dependencies = []
    for arg in flat_args_list:
        if isinstance(arg, str):
            dependencies.extend(_get_dependencies_from_graph(tool_idx, tool_name, arg))
    
    return dependencies


def convert_annotation_to_dicts(annotations: list[dict[str]]) -> dict[int, dict[str]]:
    task_dict = {
        idx: {
            "tool_name": task["tool_name"],
            "tool_args": task["tool_args"],
            "dependencies": get_dependencies(idx, task["tool_name"], task["tool_args"])
        }
        for idx, task in enumerate(annotations, start=1)
    }
    return task_dict
