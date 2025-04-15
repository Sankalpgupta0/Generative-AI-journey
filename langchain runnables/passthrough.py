from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_output_tokens=1024,
)

prompt1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, llm, parser)

prompt2 = PromptTemplate(
    template="explain this joke : {joke}",
    input_variables=["joke"],
)

joke_exp_chain = RunnableSequence(prompt2, llm, parser)

passthrough = RunnablePassthrough()

parallel_chain = RunnableParallel({
    'joke' : passthrough,
    'explanation' : joke_exp_chain,
})

chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain,
)
print(chain.invoke({"topic": "cricket"}))