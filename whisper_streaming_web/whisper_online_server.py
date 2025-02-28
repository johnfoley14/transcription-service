import asyncio
import websockets
import numpy as np
import ffmpeg
import argparse
import logging
import math                    
import json
from time import time, sleep
from datetime import timedelta
from contextlib import asynccontextmanager

from src.whisper_streaming.whisper_online import backend_factory, online_factory, add_shared_args
from src.whisper_streaming.timed_objects import ASRToken

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Parse Arguments
parser = argparse.ArgumentParser(description="Whisper WebSocket Server")
parser.add_argument("--host", type=str, default="localhost", help="Server host address.")
parser.add_argument("--port", type=int, default=8000, help="Server port.")
parser.add_argument("--transcription", type=bool, default=True, help="Enable live transcription.")

add_shared_args(parser)
args = parser.parse_args()

# Audio Constants
SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # s16le = 2 bytes per sample
MAX_BYTES_PER_SEC = 32000 * 5  # Buffer up to 5 seconds of audio

# Initialize ASR and Diarization
if args.transcription:
    asr, tokenizer = backend_factory(args)
else:
    asr, tokenizer = None, None

async def start_ffmpeg_decoder():
    """
    Start an FFmpeg process in async mode to decode WebM to raw PCM.
    """
    process = (
        ffmpeg.input("pipe:0", format="webm")
        .output("pipe:1", format="s16le", acodec="pcm_s16le", ac=CHANNELS, ar=str(SAMPLE_RATE))
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    return process

async def handle_client(websocket):
    logger.info("New WebSocket connection.")

    ffmpeg_process = None
    pcm_buffer = bytearray()
    online = online_factory(args, asr, tokenizer) if args.transcription else None

    async def restart_ffmpeg():
        nonlocal ffmpeg_process, online, pcm_buffer
        if ffmpeg_process:
            try:
                ffmpeg_process.kill()
                await asyncio.get_event_loop().run_in_executor(None, ffmpeg_process.wait)
            except Exception as e:
                logger.warning(f"Error killing FFmpeg process: {e}")
        ffmpeg_process = await start_ffmpeg_decoder()
        pcm_buffer = bytearray()
        online = online_factory(args, asr, tokenizer) if args.transcription else None
        logger.info("FFmpeg process restarted.")

    await restart_ffmpeg()

    async def ffmpeg_stdout_reader():
        nonlocal ffmpeg_process, online, pcm_buffer
        loop = asyncio.get_event_loop()
        full_transcription = ""
        beg = time()
        beg_loop = time()
        tokens = []

        while True:
            try:
                elapsed_time = math.floor((time() - beg) * 10) / 10  # Round to 0.1 sec
                ffmpeg_buffer_from_duration = max(int(32000 * elapsed_time), 4096)
                beg = time()

                try:
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(None, ffmpeg_process.stdout.read, ffmpeg_buffer_from_duration),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg read timeout. Restarting...")
                    await restart_ffmpeg()
                    full_transcription = ""
                    beg = time()
                    continue

                if not chunk:
                    logger.info("FFmpeg stdout closed.")
                    break

                pcm_buffer.extend(chunk)
                if len(pcm_buffer) >= MAX_BYTES_PER_SEC:
                    pcm_array = np.frombuffer(pcm_buffer[:MAX_BYTES_PER_SEC], dtype=np.int16).astype(np.float32) / 32768.0
                    pcm_buffer = pcm_buffer[MAX_BYTES_PER_SEC:]

                    if args.transcription:
                        logger.info(f"{len(online.audio_buffer) / online.SAMPLING_RATE} sec of audio processed.")
                        online.insert_audio_chunk(pcm_array)
                        new_tokens = online.process_iter()
                        tokens.extend(new_tokens)
                        full_transcription += " ".join([t.text for t in new_tokens])

                    response = {"transcription": full_transcription}
                    await websocket.send(json.dumps(response))


            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                break

        logger.info("Exiting ffmpeg_stdout_reader...")

    stdout_reader_task = asyncio.create_task(ffmpeg_stdout_reader())

    try:
        while True:
            message = await websocket.recv()
            try:
                ffmpeg_process.stdin.write(message)
                ffmpeg_process.stdin.flush()
            except (BrokenPipeError, AttributeError) as e:
                logger.warning(f"Error writing to FFmpeg: {e}. Restarting...")
                await restart_ffmpeg()
                ffmpeg_process.stdin.write(message)
                ffmpeg_process.stdin.flush()
    except websockets.exceptions.ConnectionClosed:
        logger.warning("WebSocket disconnected.")
    finally:
        stdout_reader_task.cancel()
        try:
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
        except:
            pass
        logger.info("Connection closed.")

async def main():
    server = await websockets.serve(handle_client, "172.22.128.1", args.port)

    logger.info(f"WebSocket server started at ws://{args.host}:{args.port}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
