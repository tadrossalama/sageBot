from pathlib import Path
import os
import sys
import faiss
import pickle
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import Prompt
from langchain.chat_models import ChatOpenAI
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain



def main():
    index = faiss.read_index("training.index")

    with open("faiss.pkl", "rb") as f:
        store = pickle.load(f)

    store.index = index

    with open("training/master.txt", "r") as f:
        promptTemplate = f.read()

    prompt = Prompt(template=promptTemplate, input_variables=["history", "context", "question"])

    chain = LLMChain(llm=ChatOpenAI(temperature=0.12), prompt=prompt)

    return chain, store

if __name__ == "__main__":

    st.set_page_config(page_title="speak your truth", page_icon=":robot:")
    st.header("Great Sage Toad")
    #prompt user to enter api key to run app
    with st.expander("Enter your OpenAI API key to begin"):
        st.write("If you don't have an API key, you can get one [here](https://beta.openai.com/)")
        key = st.text_input("API Key")
        os.environ["OPENAI_API_KEY"] = key

    while key != "":
        chain, store = main()

        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        def get_text():
            input_text = st.text_input("You: ", "Hello, how are you?", key="input")
            return input_text

        user_input = get_text()

        def onMessage(question, history):
            docs = store.similarity_search(question)
            contexts = []
            for i, doc in enumerate(docs):
                contexts.append(f"Context {i}:\n{doc.page_content}")
                answer = chain.predict(question=question, context="\n\n".join(contexts), history=history)
            return answer

        if user_input:
            history = [" ".join(st.session_state["past"]), user_input]
            
            result = onMessage(user_input,history)
            output = f"Answer: {result}"

            st.session_state.past.append(user_input)
            
            st.session_state.generated.append(output)

        if st.session_state["generated"]:

            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
