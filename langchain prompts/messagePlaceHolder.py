from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder (variable_name='chat_history'),
    ('human', '{query}')
])

chatHistory = []

# load chat history
with open('chatHistory.txt') as f:
    chatHistory.extend(f.readlines())
    
# invoke chat template
prompt = chat_template.invoke({'chat_history': chatHistory, 'query': 'where is my refund?'})
print(prompt)
