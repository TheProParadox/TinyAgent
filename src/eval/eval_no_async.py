import json
import asyncio
from dataclasses import dataclass
import networkx as nx
from tqdm import tqdm

from src.tiny_agent.tiny_agent import TinyAgentNoReplanning
from src.tiny_agent.config import get_tiny_agent_config
from src.eval.utils import convert_tasks_to_dicts, convert_annotation_to_dicts
from src.eval.graph_isomorphism import construct_digraph_from_dict, are_task_graphs_isomorphic


async def generate_single_plan(tiny_agent: TinyAgentNoReplanning, query: str) -> nx.DiGraph:
    tasks = await tiny_agent.arun(query=query)
    task_dict = convert_tasks_to_dicts(tasks)
    task_graph = construct_digraph_from_dict(task_dict)
    return task_graph


@dataclass
class Generation:
    generation_id: str
    task_graph: nx.DiGraph


def is_generation_correct(annotations: dict[str, dict[str]], generation: Generation) -> int:
    gt_task_dict = convert_annotation_to_dicts(annotations[generation.generation_id]["output"][0]["parsed_output"])
    gt_task_graph = construct_digraph_from_dict(gt_task_dict)
    if are_task_graphs_isomorphic(gt_task_graph, generation.task_graph)==True:
        return 1
    else:
        return 0


async def evaluate(annotations_path: str, tinyagent_config_path: str) -> float:
    with open(annotations_path, 'r') as fp:
        annotations = json.load(fp)
    
    num_datapoints = len(annotations.keys())
    num_correct = 0

    tiny_agent_config = get_tiny_agent_config(tinyagent_config_path)
    tiny_agent = TinyAgentNoReplanning(tiny_agent_config)

    for annotation_id, ground_truth in tqdm(annotations.items(), desc="Evaluation progress"):
        task_graph = await generate_single_plan(tiny_agent, ground_truth["input"])
        generation = Generation(annotation_id, task_graph)
        num_correct += is_generation_correct(annotations, generation)
    
    success_rate = num_correct/num_datapoints

    return success_rate


if __name__=="__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    import json
    from datetime import datetime
    from uuid import uuid4
    

    class runtime_input:
        annotations_path: str
        tinyagent_config_path: str
        save_dir: str

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config_path", type=str, required=True)
    args = arg_parser.parse_args()
    with open(args.config_path, 'r') as fp:
        runtime_input = json.load(fp)

    success_rate = asyncio.run(evaluate(runtime_input["annotations_path"], runtime_input["tinyagent_config_path"]))
    print(success_rate)

    save_dir = Path(runtime_input["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_filename = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_' + uuid4().hex + ".txt"
    with open(save_dir/save_filename, "w+") as fp:
        fp.writelines([
            f"Config path: {args.config_path}\n",
            f"Success rate: {success_rate}\n"])
