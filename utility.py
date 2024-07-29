from openai import OpenAI
import os
import base64
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI



api_key = st.secrets["openai_api_key"]
gmap_api = st.secrets["gmap_api"]
import googlemaps
from datetime import datetime

#google api_key

#llm 
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
# client = OpenAI(api_key=api_key)
client = OpenAI(api_key=api_key)




def my_response(a):
    prompt = PromptTemplate.from_template(
         """
Given the following JSON data{a} create a comprehensive journey guide that vividly explains the user journey in the simplest way possible by 
taking note of the important steps through out the journey. it is paramount to account for time within the journey and the important note the arrival stops and departure stops during the journey. it should take the format of a continous text. it is important to note
that bus number, train name and tram name are found withing the key "short_name". 
example
"short_name" : "501", signifies bus number 501
"""
    )
    
    # Initialize the LLM chain with the language model and the prompt
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain and get the result
    result = chain.run(a)
    return result



def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript


def fin_response(transcript):
    # Create a prompt template with a clear instruction
    prompt = PromptTemplate.from_template(
        "Given the following user question: {transcript}, extract the start location and the destination location. "
        "Return them in the format 'Start: [start_location], Destination: [destination_location]'."
    )
    
    # Initialize the LLM chain with the language model and the prompt
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain and get the result
    result = chain.run(transcript)
    
    # Extract start location and destination location from the result
    try:
        start_location = result.split('Start: ')[1].split(', Destination: ')[0].strip()
        destination_location = result.split('Destination: ')[1].strip()
        
    except (IndexError, ValueError):
        raise ValueError("The model response was not in the expected format.")
    
    return start_location, destination_location



def text_to_speech(input_text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input_text
    )
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        response.stream_to_file(webm_file_path)
    return webm_file_path

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)


#connecting to google map
maps = googlemaps.Client( key=gmap_api)
def location(a,b):
    # startDestination = input("where will you begin your drive?\n")
    startDestination = a
    EndDestination = b 

    now = datetime.now()
    Direction = maps.directions(startDestination,EndDestination,mode="transit",departure_time=now)
    if isinstance(Direction, list) and len(Direction) > 0:
        route = Direction[0]
        # print(route)
    else:
        print("No routes found.")
        route = None
    return Direction
