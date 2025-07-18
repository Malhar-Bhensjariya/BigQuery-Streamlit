import streamlit as st
from components.navigation import back_button

def show():
    st.title("🔮 Prediction")
    back_button()
    
    st.info("🚧 Vertex AI prediction functionality will be implemented here.")
    st.write("This page will handle ML predictions using Vertex AI models.")
