import abc
import os
from dataclasses import dataclass
from typing import Collection, Sequence

import torch
from typing_extensions import TypedDict

from src.tiny_agent.models import TinyAgentToolName
from src.tools.base import StructuredTool, Tool
from src.tiny_agent.tool_rag.embedder import Embedder

TOOLRAG_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ToolRAGResult:
    in_context_examples_prompt: str
    retrieved_tools_set: Collection[TinyAgentToolName]


class Example(TypedDict):
    text: str
    embedding: torch.Tensor
    tools: Sequence[str]


class BaseToolRAG(abc.ABC):
    """
    The base class for the ToolRAGs that are used to retrieve the in-context examples and tools based on the user
    query.
    """

    _EMBEDDINGS_DIR_PATH = os.path.join(TOOLRAG_DIR_PATH)
    _EMBEDDINGS_FILE_NAME = "examples.pt"

    # Embedder object encapsulates the embedding model that computes the embeddings for the examples/tools and the
    # user query
    _embedder: Embedder
    # The set of available tools so that we do an initial filtering based on the tools that are available
    _available_tools: Sequence[TinyAgentToolName]
    # The path to the examples file
    _embeddings_path: str

    def __init__(self, embedder: Embedder, tools: Sequence[Tool | StructuredTool]) -> None:
        self._embedder = embedder
        self._available_tools = [TinyAgentToolName(tool.name) for tool in tools]
        
        _embeddings_path = os.path.join(
            BaseToolRAG._EMBEDDINGS_DIR_PATH,
            embedder.model_name,
            BaseToolRAG._EMBEDDINGS_FILE_NAME,
        )
        assert os.path.exists(_embeddings_path), ("When using ToolRAG, a precomputed embeddings file for "
                                                  "in-context examples must be present at the path "
                                                  f"{_embeddings_path}")
        self._embeddings_path = _embeddings_path

    @property
    @abc.abstractmethod
    def tool_rag_type(self) -> str:
        pass

    @abc.abstractmethod
    def retrieve_examples_and_tools(self, query: str, top_k: int) -> ToolRAGResult:
        """
        Returns the in-context examples as a formatted prompt and the tools that are relevant to the query.
        """
        pass

    def _retrieve_top_k_embeddings(
        self, query: str, examples: list[Example], top_k: int
    ) -> list[Example]:
        """
        Computes the cosine similarity of each example and retrieves the closest top_k examples.
        If there are already less than top_k examples, returns the examples directly.
        """
        if len(examples) <= top_k:
            return examples

        query_embedding = self._embedder.embed_query(query)
        embeddings = torch.stack(
            [x["embedding"] for x in examples]
        )  # Stacking for batch processing

        # Cosine similarity between query_embedding and all chunks
        cosine_similarities = torch.nn.functional.cosine_similarity(
            embeddings, query_embedding.unsqueeze(0), dim=1
        )

        # Retrieve the top k indices from cosine_similarities
        _, top_k_indices = torch.topk(cosine_similarities, top_k)

        # Select the chunks corresponding to the top k indices
        selected_examples = [examples[i] for i in top_k_indices]

        return selected_examples

    def _load_filtered_embeddings(self, filter_tools: list[TinyAgentToolName] | None = None) -> list[Example]:
        """
        Loads the examples file that contains a list of Example objects and returns the filtered results based
        on the available tools.
        """

        embeddings: dict[str, Example] = torch.load(self._embeddings_path, weights_only=True)
        filtered_embeddings = []
        tool_names = [tool.value for tool in filter_tools or self._available_tools]
        for embedding in embeddings.values():
            # Check if all tools are available in this example
            if all(tool in tool_names for tool in embedding["tools"]):
                filtered_embeddings.append(embedding)

        return filtered_embeddings

    @staticmethod
    def _get_in_context_examples_prompt(embeddings: list[Example]) -> str:
        examples = [example["text"] for example in embeddings]
        examples_prompt = "###\n".join(examples)
        return f"{examples_prompt}###\n"
