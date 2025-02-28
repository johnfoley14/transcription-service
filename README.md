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