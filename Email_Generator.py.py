from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))




recipient = input("Who is the recipent?")
situation = input("what is the situation?")
tone = input("what tone should the email have?")

print ("\n----Email Generator----")
print("Recipent:", recipient)
print("situation:", situation)
print("Tone:", tone)

prompt = f"""
You are a professional email writing assistant.

Your task is to generate a complete professional email based on
the recipient, situation, and tone provided by the user.

Here are examples showing the expected style and structure.

Example 1:

Input:
Recipient: Professor
Situation: Request an assignment extension
Tone: Respectful

Output:
Subject: Request for Assignment Extension

Dear Professor,

I am writing to respectfully request an extension for my assignment
deadline. I would greatly appreciate your consideration.

Kind regards,
[Your Name]


Example 2:

Input:
Recipient: Manager
Situation: Request a meeting
Tone: Professional

Output:
Subject: Request for a Meeting

Dear Manager,

I would like to request a meeting to discuss an important matter.
Please let me know a suitable time for you.

Kind regards,
[Your Name]


Example 3:

Input:
Recipient: Client
Situation: Apologize for a project delay
Tone: Formal

Output:
Subject: Apology for Project Delay

Dear Client,

I sincerely apologize for the delay in the project. We are working
to resolve the situation and complete the project as soon as possible.

Kind regards,
[Your Name]


Now generate an email for the following input:

Recipient: {recipient}
Situation: {situation}
Tone: {tone}

Output format:

Subject: [email subject]

Email:
[greeting]

[email body]

[closing]
"""

response = client.chat.completions.create(
    model = "openai/gpt-oss-120b",
    messages = [
        {
            "role":"user",
            "content":prompt
        }

    ]
)

email = response.choices[0].message.content
print("\n--- Generated Email ---")
print(email)

