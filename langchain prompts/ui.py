from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.header('Research Assistant')

paper_input = st. selectbox( "Select Research Paper Name", ["Select...", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANS on Image Synthesis"] )
style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )
length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium(3-5 paragraphs)", "Long (detailed explanation)"] )

# template 
template = PromptTemplate(
    template="""
        Please summarize the research paper titled "{paper_input}" with the following specifications:
        Explanation Style: {style_input}
        Explanation Length: {length_input}
        1. Mathematical Details:
        - Include relevant mathematical equations if present in the paper.
        Explain the mathematical concepts using simple, intuitive code snippets where
        applicable.
        2. Analogies:
        Use relatable analogies to simplify complex ideas.
        Icertain information is not available in the paper, respond with: "Insufficient
        information available instead of guessing.
        Ensure the summary is clear, accurate, and aligned with the provided style and length.|
    """,
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True,
)

# fill placeholders
prompt = template.invoke({
    "paper_input" : paper_input,
    "style_input" : style_input,
    "length_input" : length_input,
})

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=1.0,
    max_completion_tokens=2000,
)

if st.button("summarize"):
    result = model.invoke(prompt)
    st.write(result.content)
