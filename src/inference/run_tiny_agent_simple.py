if __name__=="__main__":
    import argparse
    import asyncio

    from src.tiny_agent.tiny_agent import TinyAgentNoReplanning
    from src.tiny_agent.config import get_tiny_agent_config

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_path", type=str, required=True)
    args = arg_parser.parse_args()
    config_path = args.config_path

    tiny_agent_config = get_tiny_agent_config(config_path=config_path)
    tiny_agent = TinyAgentNoReplanning(tiny_agent_config)

    response = asyncio.run(
        tiny_agent.arun(query="Create a meeting with Sid and Lutfi for tomorrow 2pm to discuss the meeting notes.")
        )
    print(response)
