import asyncio
import websockets

async def send_audio():
    async with websockets.connect("ws://localhost:8000") as ws:
        with open("sample_audio.webm", "rb") as f:
            while chunk := f.read(4096):
                await ws.send(chunk)
        response = await ws.recv()
        print("Received:", response)

asyncio.run(send_audio())
