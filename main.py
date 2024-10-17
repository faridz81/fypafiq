import os
import pandas as pd
import streamlit as st
from io import StringIO
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_csv_agent
import requests

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Define the prompt template
PROMPT_TEMPLATE = (
    "You are a data assistant. Your task is to assist users in retrieving information from the provided JSON data."
    "Data is about attendance log of the student of my class. My class means that I am a lecturer. Each one json object contains one attendance of a student."
    "Count number of object with same student name, to count total attendance for each student."
    "IMPORTANT! You determine a class is distinct based on class_name. Do not use partial match for class name."
    "Use date and time in easy readable format, for example 20 October  2024 12:00 PM."
    "Different punched date and time is consider different attendance, although student attend the same class name."
    "Answer with natural language, don't use json code or other code as answer. Answer in Malay if question in Malay. Answer in English if Question in English. Express count number by digit not text.  Explain your answer. Be friendly."
    "DO not always use table. If data more than two column use table to show data. If not, use bullet" 
)

def main():
    st.set_page_config(page_title="ASK YOUR CSV")
    st.header("ASK YOUR CSV")

    # Get the filename from the URL parameters
    query_params = st.query_params
    filename = query_params.get("file")

    if filename:
        # Construct the URL to fetch the CSV file
        url = f"https://fyp.smartsolah.com/{filename}"

        # Try to load the CSV file from the URL
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error if the request fails
            csv_data = StringIO(response.text)
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading CSV file: {e}")
            return

        # Create the agent with the loaded CSV
        try:
            agent = create_csv_agent(
                ChatGroq(
                    model="llama-3.1-70b-versatile",
                    temperature=0
                ),
                csv_data,
                verbose=True,
                handle_parsing_errors=True,
                allow_dangerous_code=True
                
            )
        except Exception as e:
            st.error(f"Error creating the agent: {e}")
            return

        # Input for user's question
        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question:
            # Include the prompt in the run method
            full_prompt = f"{PROMPT_TEMPLATE}\n\nUser Question: {user_question}"
            with st.spinner(text="In progress..."):
                try:
                    response = agent.run(full_prompt)
                    st.write(response)
                except Exception as e:
                    st.error(f"Error in generating the response: {e}")

    else:
        st.warning("Please provide a 'file' parameter in the URL.")


if __name__ == "__main__":
    main()
