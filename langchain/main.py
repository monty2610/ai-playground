from dotenv import load_dotenv
import getpass
import os

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq:")

from langchain.chat_models import init_chat_model

model = init_chat_model("llama3-8b-8192", model_provider="groq")

from langchain_core.messages import HumanMessage, SystemMessage

messages = [SystemMessage("Translate this message from English to Hindi"), HumanMessage("Hi!")]

response = model.invoke(messages)

print(response)

