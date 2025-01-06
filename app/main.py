from whisper import whisper
from datetime import datetime

def translate_audio(file_path):
    model = whisper.load_model("tiny.en")
    result = model.transcribe(file_path)
    return result['text']

time = datetime.now()
translated_text = translate_audio("./Recording1.wav")
print(datetime.now() - time)
print(translated_text)