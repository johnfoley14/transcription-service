#!/usr/bin/env python3
import sys, argparse, os, logging, asyncio, websockets, numpy as np, sounddevice as sd, signal, json
from whisper_online import *
from datetime import datetime

# Setup
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument("--warmup-file", type=str, dest="warmup_file")
add_shared_args(parser)
args = parser.parse_args()
set_logging(args, logger)

# Whisper Setup

SAMPLING_RATE = 16000
logger.info(f"Args: {args}")
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size

# Warmup
msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file, 0, 1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. " + msg)
        sys.exit(1)
else:
    logger.warning(msg)

# Transcription formatting
last_end = None
def format_output_transcript(o):
    global last_end
    if o[0] is not None:
        beg, end = o[0]*1000, o[1]*1000
        if last_end is not None:
            beg = max(beg, last_end)
        last_end = end

        return {
                "start":round(beg, 0),
                "end":round(end, 0),
                "transcript":o[2]}
    return None

# WebSocket handler
async def transcribe_websocket(websocket):
    global last_end
    connection_start = datetime.now().timestamp() * 1000
    logger.info("Client connected.")
    buffer = []
    online.init() # Initiate Whisper online VAC processor
    running = True

    def audio_callback(indata, frames, time, status):
        if status:
            logger.info("Stream status:", status, file=sys.stderr)
        buffer.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        blocksize=int(SAMPLING_RATE * 0.25)
    )

    try:
        with stream:
            while running:
                await asyncio.sleep(0.1)
                total_len = sum(len(chunk) for chunk in buffer)
                if total_len >= SAMPLING_RATE * min_chunk:
                    audio_data = np.concatenate(buffer, axis=0).flatten()
                    buffer.clear()

                    online.insert_audio_chunk(audio_data)
                    o = online.process_iter()
                    result = format_output_transcript(o)
                    if result:
                        result["connection start"] = connection_start
                        await websocket.send(json.dumps(result))
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.exception(f"Error during transcription: {e}")
    finally:
        running = False
        stream.close()
        logger.info("Audio stream closed.")

# Start WebSocket server
async def main():
    host = "localhost"
    port = 8765
    logger.info(f"WebSocket STT server listening on ws://{host}:{port}")
    async with websockets.serve(transcribe_websocket, host, port):
        await asyncio.Future()  # Run forever

# Support CTRL+C to exit
def handle_sigint(sig, frame):
    logger.info("Shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

if __name__ == "__main__":
    asyncio.run(main())
