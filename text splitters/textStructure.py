# \n\n -> paragraph
# \n -> line change
# ' ' -> space / word
# '.' -> character based split

from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """Artificial Intelligence is transforming the way we interact with technology. From personalized recommendations to autonomous vehicles.
AI is at the core of modern innovation. Its ability to analyze massive amounts of data and make intelligent decisions is revolutionizing industries across the globe. As AI continues to evolve, its impact on society will only grow deeper and more profound. However, with great power comes great responsibility. Ethical considerations surrounding AI, such as bias in algorithms and data privacy, are critical to address as we move forward.
The future of AI holds immense potential, but it is essential to navigate its challenges thoughtfully and responsibly. By fostering collaboration between technologists, ethicists, and policymakers, we can ensure that AI serves humanity's best interests and contributes positively to our world.
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

result = text_splitter.split_text(text)
print(result)