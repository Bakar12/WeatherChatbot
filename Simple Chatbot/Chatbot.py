import random
import spacy
import requests
from transformers import pipeline
from datetime import datetime, timedelta
import json

# Load the spaCy model for named entity recognition
nlp_spacy = spacy.load("en_core_web_sm")
# Load the pre-trained model for question answering
nlp_transformers = pipeline("question-answering")

# Load the configuration file
with open('Simple%20Chatbot/config.json') as f:
    config = json.load(f)

# Get the API key from the configuration
api_key = config['api_key']
BASE_URL = 'http://api.openweathermap.org/data/2.5/'
HISTORICAL_BASE_URL = 'http://history.openweathermap.org/data/2.5/history/city'
CACHE_FILE = 'weather_cache.json'
CACHE_EXPIRY = timedelta(minutes=30)  # Cache expiry time

# Define a dictionary of intents and responses
intents = {
    "greeting": ["Hello!", "Hi there!", "Hey! How can I help you?"],
    "goodbye": ["Goodbye!", "See you later!", "Take care!"],
    "thank_you": ["You're welcome!", "No problem!", "Glad to help!"],
    "name": ["I'm a more complex chatbot created by a Python programmer."],
    "weather": ["Please provide the city name to get the weather information."],
    "news": ["I'm sorry, I can't fetch the news right now. Try again later."]
}

# List of keywords for each intent
keywords = {
    "greeting": ["hello", "hi", "hey"],
    "goodbye": ["bye", "goodbye", "see you"],
    "thank_you": ["thank you", "thanks"],
    "name": ["your name", "who are you"],
    "weather": ["weather", "forecast"],
    "news": ["news", "headlines"]
}


def get_intent(user_input):
    for intent, words in keywords.items():
        if any(word in user_input.lower() for word in words):
            return intent
    return None


def extract_city(user_input):
    doc = nlp_spacy(user_input)
    for entity in doc.ents:
        if entity.label_ == "GPE":  # GPE (Geo-Political Entity) includes cities
            return entity.text
    return None


def get_weather(city):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city
    response = requests.get(complete_url)
    data = response.json()
    if data["cod"] != "404":
        main = data["main"]
        temperature = main["temp"]
        weather_desc = data["weather"][0]["description"]
        return f"The temperature in {city} is {temperature - 273.15:.2f}°C with {weather_desc}."
    else:
        return "City not found."


context = {}


def update_context(user_input, intent):
    context["previous_intent"] = intent
    context["previous_input"] = user_input


def chatbot_response(user_input):
    intent = get_intent(user_input)
    update_context(user_input, intent)
    if intent == "weather":
        city = extract_city(user_input)
        if city:
            return get_weather(city)
        else:
            return "Please specify a city name."
    elif intent:
        return random.choice(intents[intent])
    else:
        return "I'm sorry, I don't understand that."


def chat():
    print("Chatbot: Hello! I'm a more complex chatbot. How can I help you?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")


if __name__ == "__main__":
    chat()
