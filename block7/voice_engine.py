import pyttsx3
import threading

def _speak_thread(text):

    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Помилка голосу: {e}")

def speak(text):
    t = threading.Thread(target=_speak_thread, args=(text,))
    t.start()