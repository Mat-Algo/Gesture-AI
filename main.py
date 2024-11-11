import google.generativeai as genai
from Modelling.speak import speak
from dotenv import load_dotenv
from os import getenv

load_dotenv()

KEY1 = getenv("KEY1AM")
genai.configure(api_key=KEY1)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
conversation_history = []

def get_response(user_message):
    """
    Gets a response from Gemini while maintaining conversation history.
    """
    global conversation_history
    conversation_history.append({"role": "user", "content": user_message})
    prompt = ""
    for turn in conversation_history:
        prompt += f"{turn['role']}: {turn['content']}\n"
    prompt += "assistant: "
    response = model.generate_content(prompt)
    conversation_history.append({"role": "assistant", "content": response.text})
    return response.text

print("Welcome to the Airport Virtual Navigation Assistant!")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        speak("Goodbye! Have a safe flight!")
        break
    assistant_response = get_response(user_input+" (You are a assistant at random airport, give arbitrary Answers like you know everything but Give a very concise,formal,and informative response as an airport assistant would)")
    print(f"Assistant: {assistant_response}")
    speak(assistant_response) # Uncomment this line to enable text-to-speech