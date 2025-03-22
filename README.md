# translation-service
A local Python backend application to provide translation services. Uses STT and TTS models and communicates with a web application via web sockets. 


# Running the Transcription Service
python -m venv venv
source venv/bin/activate (Unix based)
venv\Scripts\activate (Windows)

pip list (will show currently installed packages in that virtual environment)

deactivate (deactive virtual environment)

install all dependencies and libraries in requirements.txt

python3 whisper_online_server.py --model small --min-chunk-size 1 --port 3000


# Fine Tune Min min_silence_duration_ms in Silero VAD iterator

This value determines the time between different of of no audio to separate different segments


# Ubuntu command to pipe raw audio to server
arecord -f S16_LE -c1 -r 16000 -t raw -D default | \
ffmpeg -f s16le -ar 16000 -ac 1 -i pipe:0 -c:a libopus -f webm pipe:1 | \
websocat -b ws://172.22.128.1:8000


ffmpeg -f dshow -i audio="Microphone Array (Intel® Smart Sound Technology for Digital Microphones)" -ac 1 -ar 16000 -f s32le - | ncat localhost 43007

cd Development/stt/whisper_streaming && python3 whisper_online_server.py --model small --min-chunk-size 1 --port 3000

arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc localhost 3000

# Current Setup
1. Using local_samplying environment on windows
2. Run using ` python local_sampling.py --model tiny.en --min-chunk-size 1` (Note python not python3)
3. Can probably just use `python stt_web_socket_service.py` now. 
