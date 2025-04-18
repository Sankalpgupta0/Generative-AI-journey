from itertools import chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_completion_tokens=2,
)

urls = ["https://www.amazon.in/s?k=macbook+air+m4&crid=31XODDT4094CB&sprefix=mac%2Caps%2C205&ref=nb_sb_ss_ts-doa-p_1_3", "https://www.amazon.in/ASUS-Gaming-15-6-39-62cm-FA506NFR-HN045W/dp/B0CDQ5CW9V/ref=sr_1_3?crid=3QX615BWHLNZZ&dib=eyJ2IjoiMSJ9.zIkVjGP40evG_tdbzqGjbNxJI5OHwGwh2spU-lYC40sY09OXOFKNusxtOrZpvN2RHGqbc3FwFVKWD674Cce-W2zZGC7akznGn3Z0ooI4BP164ww6bp25_7bvhmIoR9jZWiF5k8woZrpauBRqpG8FvjQnTChSgqbhJAim4TUgS4JW4Fc4B2fHRP0eXR3mVMDzD8OLIHMN2Xy17RxnzpAbErBlIhzP0i82h1riopEXliE.P3PkYULTR2kgnPdGxIiCneuzysk9HVh8a6ug1FfvhSw&dib_tag=se&keywords=laptop+under+60k&qid=1744894626&sprefix=lapt%2Caps%2C216&sr=8-3"]
loader = WebBaseLoader(urls)

docs = loader.load()

lapout1 = (docs[0].page_content).replace('\n', '')
lapout2 = (docs[1].page_content).replace('\n', '')

prompt = PromptTemplate(
    template="make a comparision between the following two laptops: {laptop1} and {laptop2}",
    input_variables=["laptop1", "laptop2"],
)

chain = prompt | model | StrOutputParser()
print(chain.invoke({'laptop1': lapout1, 'laptop2': lapout2}))