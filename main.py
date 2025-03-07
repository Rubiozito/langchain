from openai import OpenAI
from dotenv import load_dotenv

api_key = load_dotenv().get('OPENAI_API_KEY')

client = OpenAI(api_key)

client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the meaning of life?"}
    ])