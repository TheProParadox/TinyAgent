import asyncio
from http import HTTPStatus

from fastapi import HTTPException
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.tiny_agent.config import get_tiny_agent_config
from src.tiny_agent.models import (
    LLM_ERROR_TOKEN,
    streaming_queue,
)
from src.tiny_agent.tiny_agent import TinyAgentNoReplanning
from src.utils.logger_utils import enable_logging, enable_logging_to_file, log

enable_logging(True)
enable_logging_to_file(False)


def empty_queue(q: asyncio.Queue) -> None:
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
        except asyncio.QueueEmpty:
            # Handle the case where the queue is already empty
            break


class TinyAgentRequest(BaseModel):
    query: str


async def execute_command(request: TinyAgentRequest, config_path) -> StreamingResponse:
    """
    This is the main endpoint that calls the TinyAgent to generate a response to the given query.
    """
    log(f"\n\n====\nReceived request: {request.query}")

    # First, ensure the queue is empty
    empty_queue(streaming_queue)

    query = request.query

    if not query or len(query) <= 0:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="No query provided"
        )

    try:
        tiny_agent_config = get_tiny_agent_config(config_path=config_path)
        tiny_agent = TinyAgentNoReplanning(tiny_agent_config)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error: {e}",
        )

    async def generate():
        try:
            response_task = asyncio.create_task(tiny_agent.arun(query))

            while True:
                # Await a small timeout to periodically check if the task is done
                try:
                    token = await asyncio.wait_for(streaming_queue.get(), timeout=1.0)
                    if token is None:
                        break
                    if token.startswith(LLM_ERROR_TOKEN):
                        raise Exception(token[len(LLM_ERROR_TOKEN) :])
                    yield token
                except asyncio.TimeoutError:
                    pass  # No new token, check task status

                # Check if the task is done to handle any potential exception
                if response_task.done():
                    break

            # Task created with asyncio.create_task() do not propagate exceptions
            # to the calling context. Instead, the exception remains encapsulated within
            # the task object itself until the task is awaited or its result is explicitly retrieved.
            # Hence, we check here if the task has an exception set by awaiting it, which will
            # raise the exception if it exists. If it doesn't, we just yield the result.
            await response_task
            response = response_task.result()
            yield f"\n\n{response}"
        except Exception as e:
            # You cannot raise HTTPExceptions in an async generator, it doesn't
            # get caught by the FastAPI exception handling middleware. Hence,
            # we are manually catching the exceptions and yielding/logging them.
            yield f"Error: {e}"
            log(f"Error: {e}")

    return generate()

async def print_response(user_query, config_path):
    async_response = await execute_command(user_query, config_path)
    async for t in async_response:
        print(t)


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config_path", type=str, required=True)
    args = arg_parser.parse_args()

    user_query = TinyAgentRequest(
        query="Create a meeting with Sid and Lutfi for tomorrow 2pm to discuss the meeting notes."
        )
    asyncio.run(print_response(user_query, args.config_path))
