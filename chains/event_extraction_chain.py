from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

prompt_template = PromptTemplate(
    input_variables=["article"],
    template=open("prompts/extract_traffic_event.txt", "r", encoding="utf-8").read()
)

event_chain = LLMChain(llm=llm, prompt=prompt_template)
