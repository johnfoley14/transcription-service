#!/usr/bin/env python3
import asyncio
import websockets
import logging
import numpy as np
import io
import soundfile

from whisper_online import *  # your existing Whisper setup
import argparse, os, sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--warmup-file", type=str, dest="warmup_file")
add_shared_args(parser)
args = parser.parse_args()

set_logging(args, logger, other="")

SAMPLING_RATE = 16000
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size

msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        logger.info("whisper is about to be warmed up")
        a = load_audio_chunk(args.warmup_file, 0, 1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. " + msg)
        sys.exit(1)
else:
    logger.warning(msg)


def format_output_transcript(o, last_end):
    """
    Format the transcript output:
    returns a string like: "0 1720 transcribed text"
    Adjusts for no overlap: beg timestamp >= last_end
    """
    if o[0] is None:
        return None, last_end  # no text
    beg, end = o[0] * 1000, o[1] * 1000
    beg = max(beg, last_end if last_end else 0)
    last_end = end
    text = f"{beg:.0f} {end:.0f} {o[2]}"
    print(text, flush=True, file=sys.stderr)
    return text, last_end


class WebSocketServerProcessor:
    def __init__(self, online_asr_proc, min_chunk_sec):
        self.online_asr_proc = online_asr_proc
        self.min_chunk_sec = min_chunk_sec
        self.buffer = bytearray()
        self.is_first = True
        self.last_end = 0

    async def process(self, websocket):
        """Receive audio chunks until enough is accumulated, then run inference."""
        self.online_asr_proc.init()
        try:
            async for message in websocket:
                # message should be raw bytes of audio
                if not isinstance(message, bytes):
                    continue  # ignore non-binary messages

                self.buffer.extend(message)

                # Convert the buffer length to sample count
                # 16-bit audio -> 2 bytes per sample -> sample_count = len(buffer)/2
                sample_count = len(self.buffer) // 2

                # If we have enough samples to exceed min_chunk_sec, run inference
                needed_samples = int(SAMPLING_RATE * self.min_chunk_sec)
                if sample_count >= needed_samples:
                    # Load from buffer
                    data_bytes = bytes(self.buffer)
                    self.buffer.clear()

                    # Convert to float32 audio
                    sf = soundfile.SoundFile(io.BytesIO(data_bytes),
                                             channels=1,
                                             endian="LITTLE",
                                             samplerate=SAMPLING_RATE,
                                             subtype="PCM_16",
                                             format="RAW")
                    audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)

                    # In case we didn't have quite enough the first time
                    if self.is_first and len(audio) < needed_samples:
                        self.buffer.extend(data_bytes)  # put it back, wait for more
                        continue

                    self.is_first = False

                    # Send chunk to the online processor
                    self.online_asr_proc.insert_audio_chunk(audio)
                    result = online.process_iter()
                    msg, self.last_end = format_output_transcript(result, self.last_end)
                    if msg:
                        await websocket.send(msg)

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error in WebSocket loop: {e}", exc_info=True)


async def ws_handler(websocket):
    """Handles each new WebSocket connection."""
    processor = WebSocketServerProcessor(online, min_chunk)
    print("New connection")
    await processor.process(websocket)


async def main():
    async with websockets.serve(ws_handler, args.host, args.port):
        logger.info(f"WebSocket server listening on {args.host}:{args.port}")
        await asyncio.Future()  # keep running

if __name__ == "__main__":
    asyncio.run(main())