import pyttsx3

def speak(text):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Set speech rate (adjust as needed)
    engine.setProperty('rate', 175)

    # Set volume (range: 0.0 to 1.0)
    engine.setProperty('volume', 1.0)

    # Choose the first voice available (adjust index as needed)
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[0].id)

    # Speak the provided text
    engine.say(text)
    engine.runAndWait()

# Test block for running this script standalone
if __name__ == "__main__":
    print("Testing speak function...")
    speak("This is a test of the text-to-speech functionality.")
    print("Test complete.")
