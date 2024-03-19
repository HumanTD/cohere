from langchain_community.embeddings.cohere import CohereEmbeddings
from langchain_community.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import streamlit as st
import flask as Flask

load_dotenv()


cohere_api_key = os.getenv("cohere_api_key")
authorization = os.getenv("authorization")
model_url=os.getenv("model_url")


def bot():
    with open('sample_resume.pdf', "rb") as f:
        loader = PyPDFLoader(f.name)
        pages = loader.load_and_split()
        return pages
    

pages=bot()

try:
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_api_key)
    store = Qdrant.from_documents(
        pages,
        embeddings,
        location=":memory:",
        collection_name="my_documents",
        distance_func="cosine",
    )
    prompt_template = """Text: {context}
    Question: {question}
    You are replying as an multilingual AI Chat Bot,
    answer the question without using any vulgularity and in maximum 50 words.Do not make your own answers and refer only to the pdf passed."""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    prompt = [{"role": "system", "content": prompt_template}]
    for message in prompt:
        if message["role"] == "user":
            print(message["content"])
        elif message["role"] == "assistant":
            print(message["content"])

    question = st.text_input(label='input')

    if question:
        prompt.append({"role": "user", "content": question})
        chain_type_kwargs = {"prompt": PROMPT}

        qa = RetrievalQA.from_chain_type(
                    llm=Cohere(model="command", temperature=0, cohere_api_key=cohere_api_key),
                    chain_type="stuff",
                    retriever=store.as_retriever(),
                    chain_type_kwargs=chain_type_kwargs,
                    return_source_documents=True,
                )
        
        answer = qa.invoke({"query": question})
        result = answer["result"].replace("\n", "").replace("Answer:", "").replace("mentioned in the text:","").replace("According to the text you provided,","").replace("According to the provided text, ","")

        lines = result.split('.')

        # Remove the last line
        updated_text = '.'.join(lines[:-1])

        updated_text=updated_text+"."

        quote_index = updated_text.find('"')

        # Extract text after the double quotes
        extracted_text = updated_text[quote_index + 1:]

        st.write(extracted_text)



except Exception as e:
    print(f"An error occurred: {str(e)}")





