from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key = os.getenv("GROQ_API_KEY")
)
messages  = [
    {"role": "system",
    "content":"you are a friendly python teacher"
    }
]

while True:

 user_input = input("You:")

 messages.append(
    {
        "role": "user",
        "content": user_input
    }
)

 print (messages)

 response = client.chat.completions.create(
    model = "openai/gpt-oss-120b",
    messages=messages
)
 assistant_message = response.choices[0].message.content
 print("Assistant:", assistant_message)

 messages.append(
    {
        "role":"assistant",
        "content": assistant_message


    }
)
 print (messages)

 exit_condition = input("Do you want to exit? (yes/no): ")
 if exit_condition.lower() == "yes":

  break
