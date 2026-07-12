import asyncio
import websockets
import json

async def dump_tosu():
    uri = "ws://localhost:24050/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to Tosu! Waiting for 1 message...")
            message = await websocket.recv()
            data = json.loads(message)
            with open("tosu_dump.json", "w") as f:
                json.dump(data, f, indent=4)
            print("Dumped data to tosu_dump.json")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(dump_tosu())
