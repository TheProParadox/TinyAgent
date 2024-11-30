from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
import torch


class Embedder:
    def __init__(self, embedding_model: OpenAIEmbeddings | AzureOpenAIEmbeddings | HuggingFaceEmbeddings,
                 examples_prefix: str = None, query_prefix: str = None) -> None:
        
        self.embedding_model = embedding_model

        if isinstance(embedding_model, AzureOpenAIEmbeddings) or isinstance(embedding_model, OpenAIEmbeddings):
            model_name = embedding_model.model
        elif isinstance(embedding_model, HuggingFaceEmbeddings):
            model_name = embedding_model.model_name
        
        self.model_name = model_name.split("/")[-1]  # Only use the last model name
        
        self.examples_prefix = examples_prefix
        self.query_prefix = query_prefix
    
    def embed_query(self, query: str) -> torch.Tensor:
        if self.query_prefix is not None:
            query = self.query_prefix + query

        query_embedding = torch.tensor(self.embedding_model.embed_query(query))

        return query_embedding
    
    def embed_example(self, example: str) -> torch.Tensor:
        if self.examples_prefix is not None:
            example = self.examples_prefix + example
        
        example_embedding = torch.tensor(self.embedding_model.embed_query(example))

        return example_embedding
