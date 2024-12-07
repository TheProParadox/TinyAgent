import json
import asyncio
from dataclasses import dataclass
from tqdm import tqdm

from src.tiny_agent.tiny_agent import TinyAgentNoReplanning
from src.tiny_agent.config import get_tiny_agent_config
from src.eval.utils import convert_tasks_to_dicts, convert_annotation_to_dicts
from src.eval.graph_isomorphism import construct_digraph_from_dict, are_task_graphs_isomorphic
from src.llm_compiler.task_fetching_unit import Task


async def generate_single_plan(tiny_agent: TinyAgentNoReplanning, query: str) -> dict[int, Task]:
    tasks = await tiny_agent.arun(query=query)
    return tasks


@dataclass
class Annotation:
    annotation_id: str
    ground_truth: dict[str]


@dataclass
class Generation:
    generation_id: str
    tasks: dict[int, Task]


class ShutdownQueue:
    pass


async def generate_plans(tiny_agent: TinyAgentNoReplanning, annotations_queue: asyncio.Queue[Annotation],
                         generation_queue: asyncio.Queue[Generation|ShutdownQueue]) -> None:
    while True:
        try:
            # Producer should not wait on a get since nothing adds to annotations_queue
            annotation = annotations_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        tasks = await generate_single_plan(tiny_agent, annotation.ground_truth["input"])
        await generation_queue.put(Generation(generation_id=annotation.annotation_id, tasks=tasks))
        annotations_queue.task_done()
    
    await generation_queue.put(ShutdownQueue())


def is_generation_correct(annotations: dict[str, dict[str]], generation: Generation) -> int:
    gt_task_dict = convert_annotation_to_dicts(annotations[generation.generation_id]["output"][0]["parsed_output"])
    gt_task_graph = construct_digraph_from_dict(gt_task_dict)

    predicted_task_dict = convert_tasks_to_dicts(generation.tasks)
    predicted_task_graph = construct_digraph_from_dict(predicted_task_dict)

    if are_task_graphs_isomorphic(gt_task_graph, predicted_task_graph)==True:
        return 1
    else:
        return 0


async def evaluate_generated_plans(generation_queue: asyncio.Queue[Generation|ShutdownQueue],
                                   annotations_queue: asyncio.Queue[Annotation], annotations: dict[str, dict[str]]
                                   ) -> int:
    num_correct = 0
    while True:
        generation = await generation_queue.get()
        if isinstance(generation, ShutdownQueue):
            # Task done is for the ShutdownQueue
            generation_queue.task_done()
            break
        num_correct += is_generation_correct(annotations, generation)
        generation_queue.task_done()

    # Shutdown procedure
    await annotations_queue.join()
    while True:
        try:
            generation = generation_queue.get_nowait()
            if not isinstance(generation, ShutdownQueue):
                num_correct += is_generation_correct(annotations, generation)
            generation_queue.task_done()

        except asyncio.QueueEmpty:
            # Put shutdown back so other evaluators can also exit. But there could be many evaluators who come to this
            # point and fill up the queue with shutdowns. Since this evaluator went through the whole queue and
            # processed every valid generation left it can just exit if the queue is full
            try:
                generation_queue.put_nowait(ShutdownQueue())
            except asyncio.QueueFull:
                pass
            return num_correct
        

async def update_progress_bar(annotations_queue: asyncio.Queue[Annotation], num_datapoints: int) -> None:
    progress_bar = tqdm(range(num_datapoints), desc="Evaluation progress")
    prev_size = num_datapoints
    while annotations_queue.empty()==False:
        cur_size = annotations_queue.qsize()
        progress_bar.update(prev_size-cur_size)
        prev_size = cur_size
        await asyncio.sleep(10)


async def evaluate(annotations_path: str, num_generators: int, num_evaluators: int, generation_queue_size: int,
                   tinyagent_config_path: str) -> float:
    with open(annotations_path, 'r') as fp:
        annotations = json.load(fp)
    
    num_datapoints = len(annotations.keys())
    num_correct = 0

    annotations_queue = asyncio.Queue(maxsize=num_datapoints)
    for annotation_id, ground_truth in annotations.items():
        annotations_queue.put_nowait(Annotation(annotation_id, ground_truth))
    
    generation_queue = asyncio.Queue(maxsize=generation_queue_size)

    tiny_agent_config = get_tiny_agent_config(tinyagent_config_path)
    tiny_agent = TinyAgentNoReplanning(tiny_agent_config)

    generators = [
        asyncio.create_task(generate_plans(tiny_agent, annotations_queue, generation_queue))
        for _ in range(num_generators)
    ]
    evaluators = [
        asyncio.create_task(evaluate_generated_plans(generation_queue, annotations_queue, annotations))
        for _ in range(num_evaluators)
    ]

    progress_bar_updater = asyncio.create_task(update_progress_bar(annotations_queue, num_datapoints))

    for evaluator in evaluators:
        num_correct += await evaluator

    for generator in generators:
        generator.cancel()
    
    progress_bar_updater.cancel()
    
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
        num_generators: int
        num_evaluators: int
        generation_queue_size: int
        tinyagent_config_path: str
        save_dir: str

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config_path", type=str, required=True)
    args = arg_parser.parse_args()
    with open(args.config_path, 'r') as fp:
        runtime_input = json.load(fp)

    success_rate = asyncio.run(evaluate(runtime_input["annotations_path"], runtime_input["num_generators"],
                                        runtime_input["num_evaluators"], runtime_input["generation_queue_size"],
                                        runtime_input["tinyagent_config_path"]))
    print(success_rate)

    save_dir = Path(runtime_input["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_filename = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_' + uuid4().hex + ".txt"
    with open(save_dir/save_filename, "w+") as fp:
        fp.writelines([
            f"Config path: {args.config_path}\n",
            f"Success rate: {success_rate}\n"])
