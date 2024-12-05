import json
import asyncio
import websockets
import os

from model.boid import LlmInput, LlmOutput
from dotenv import load_dotenv
from src.config import Config
from src.desire import get_desire
from src.rules import generate_rules, encode_rules

load_dotenv()

async def websocket_client():
    uri = f"ws://{os.getenv('WEBSOCKET_URL')}"
    try:
        async with websockets.connect(uri, additional_headers={"identification": "boid-llm"}) as websocket:
            print(f"Connected to WebSocket server: {uri}")
            while True:
                # Receive raw binary data
                message = await websocket.recv()
                print("Message received from server.")

                # Deserialize the Protobuf message
                llm_input = LlmInput().parse(message)

                desires = get_desire()
                rules = generate_rules(desires)
                proto_rules = encode_rules(rules)

                await websocket.send(json.dumps({ "event": "rules", "data": list(LlmOutput(rules=proto_rules, user_id=llm_input.user.id).SerializeToString()) }))

                # Access fields in the Protobuf message
                print(f"Received Protobuf Message: {llm_input}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the client
if __name__ == "__main__":
    asyncio.run(websocket_client())

