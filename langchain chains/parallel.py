from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    temperature=0,
    model="gemini-1.5-pro",
    max_output_tokens=512,
)

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
)

# model2 = ChatHuggingFace(
#     llm=llm,
# )

model2 = model1

prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Make a detailed report on {topic}. upto 2000 words.",
)

prompt2 = PromptTemplate(
    input_variables=["report"],
    template="generate short and simple notes form this {report}",
)

prompt3 = PromptTemplate(
    input_variables=["report"],
    template="generate 5 MCQs from this {report}",
)

prompt4 = PromptTemplate(
    input_variables=["notes", "mcqs"],
    template="merge the provided notes -> {notes} and MCQs -> {mcqs} into a single document.",
)

parser = StrOutputParser()

report = prompt1 | model1 | parser

parallel_chain = RunnableParallel({
    'notes' : prompt2 | model2 | parser,
    'mcqs' : prompt3 | model1 | parser,
})

merge_chain = prompt4 | model1 | parser

chain = report | parallel_chain | merge_chain

result = chain.invoke({"topic": "Artificial Intelligence"})

# print(result)

chain.get_graph().print_ascii()