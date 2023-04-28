import pickle
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

sbert_model=joblib.load('sbert.pkl')

with open('resume_embeds.pkl', 'rb') as f:
    df = pickle.load(f)


st.title('Resume Shortlisting Tool')

# Define the input field for skills

skills_input = st.text_input('Enter the desired skills (comma-separated)python, data analysis, machine learning')


def resume_suggestion(skills_input):

    JD_skills = sbert_model.encode(skills_input)
    paths = df[(cosine_similarity(list(df['skill_embeds']),[JD_skills]) > 0.85)][['path','skills']]
    paths=paths.reset_index(drop=True)
    return paths


button=st.button("Submit")


if button:
    paths = resume_suggestion(skills_input)
    st.dataframe(paths)