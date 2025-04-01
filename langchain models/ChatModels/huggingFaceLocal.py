from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

# Create a text-generation pipeline
hf_pipeline = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

# Wrap the pipeline for LangChain (pass max_new_tokens inside pipeline_kwargs)
llm = HuggingFacePipeline(
    pipeline=hf_pipeline, 
    pipeline_kwargs={
        "max_new_tokens": 2000, 
        "temperature": 0.8
    }
)

# Use ChatHuggingFace
model = ChatHuggingFace(
    llm=llm
)

# Invoke the model
result = model.invoke("What is the capital of India?")
print("\n", result.content)
