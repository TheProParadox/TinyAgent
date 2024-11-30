import torch
import os
from tqdm import tqdm
import pathlib


from src.tiny_agent.tool_rag.embedder import Embedder
from src.tiny_agent.tool_rag.base_tool_rag import Example


TOOLRAG_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR_PATH = pathlib.Path(TOOLRAG_DIR_PATH)


def create_example_embeddings(embedder: Embedder, examples: dict[str, Example]):
    
    for example_id in tqdm(examples.keys(), desc="Embedding examples"):
        example_text = examples[example_id]["text"]
        example_embedding = embedder.embed_example(example_text)
        examples[example_id]["embedding"] = example_embedding

    save_dir = EMBEDDINGS_DIR_PATH/f"{embedder.model_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir/"examples.pt"
    torch.save(examples, save_path)


if __name__=="__main__":
    from argparse import ArgumentParser
    import json

    from src.utils.model_utils import get_embedder
    from src.tiny_agent.models import AgentType
    from src.tiny_agent.config import get_model_config, load_config


    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config_path", type=str, required=True)
    args = arg_parser.parse_args()

    config = load_config(args.config_path)
    embedding_model_config = get_model_config(config, config["toolRAGProvider"], AgentType.EMBEDDING)
    embedder = get_embedder(
                model_type=embedding_model_config.model_type.value,
                model_name=embedding_model_config.model_name,
                api_key=embedding_model_config.api_key,
                azure_endpoint=config.get("azure_endpoint"),
                azure_embedding_deployment=embedding_model_config.model_name,
                azure_api_version=config.get("azure_api_version"),
                local_port=embedding_model_config.port,
                context_length=embedding_model_config.context_length,
                hf_trust_remote_code=embedding_model_config.hfTrustRemoteCode,
                examples_prefix=embedding_model_config.examplesPrefix,
                query_prefix=embedding_model_config.queryPrefix
            )
    
    with open(EMBEDDINGS_DIR_PATH/"examples.json", 'r') as fp:
        examples = json.load(fp)

    create_example_embeddings(embedder, examples)
