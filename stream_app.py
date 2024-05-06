import streamlit as st
from st_nlp_medical_qa import return_to_streamlit

st.header("Medical QA") # title of the app
st.text("Suggesting similar questions to those asked by the user")

user_question = st.text_input("Enter your question:")

click = st.button("Submit")

if click:
    st.write(return_to_streamlit(user_question))

