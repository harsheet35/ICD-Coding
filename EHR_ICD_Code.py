from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import openai

load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')
print(api_key)
headers = {
    "content-type":"application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I provide a prompt to the GPT-3.5 Turbo model using the API?"}
    ]
}

# response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

# print(response.json()['choices'][0]['message']['content'])

# from openai import OpenAI
# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )

# print(completion.choices[0].message)
openai.api_key = api_key

# List available models
models = openai.models.list()

# Extract model IDs
# model_ids = [model["id"] for model in models["data"]]

# Print the model IDs
# for model_id in model_ids:
    # print(model_id)
print(models)