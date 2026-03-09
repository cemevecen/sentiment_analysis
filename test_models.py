import os
from dotenv import load_dotenv
from google import genai
import sys

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash-lite',
        contents='Merhaba',
    )
    print("lite:", response.text)
except Exception as e:
    print("lite error:", e)

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Merhaba',
    )
    print("2.5 flash:", response.text)
except Exception as e:
    print("2.5 flash error:", e)

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents='Merhaba',
    )
    print("1.5 flash:", response.text)
except Exception as e:
    print("1.5 flash error:", e)
