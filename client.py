import requests
import streamlit as st

def get_angry_response(input_text1):
    response = requests.post("http://localhost:8000/angry/invoke",
    json={'input':{"topic":input_text1}})
    print(response)
    return response.json()['output']


def get_funny_response(input_text2):
    response = requests.post("http://localhost:8000/funny/invoke",
                              json={'input':{"topic":input_text2}})
    return response.json()['output']

st.title("Dual Tone LLM response")
input_text1 = st.text_input("Angry Response Topic")
input_text2 = st.text_input("Funny Response Topic")

if input_text1:
    st.write(get_angry_response(input_text1))

if input_text2:
    st.write(get_funny_response(input_text2))