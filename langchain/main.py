from dotenv import load_dotenv
import getpass
import os

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq:")

from langchain.chat_models import init_chat_model

model = init_chat_model("llama3-8b-8192", model_provider="groq")

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate following from english into {language}"

prompt_template = ChatPromptTemplate.from_messages([("system", system_template), ("user", "{text}")])

prompt = prompt_template.invoke({ "language": "hindi", "text": "My name is Naman Jain"})


response = model.invoke(prompt)

print(response.content)

