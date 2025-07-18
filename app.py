import streamlit as st
from components.navigation import init_state
from pages import (
    Dataset_Selection,
    Table_Selection,
    File_Upload,
    Prediction
)

# Page configuration
st.set_page_config(
    page_title="BigQuery Streamlit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stInfo {
        border-radius: 8px;
    }
    .stSuccess {
        border-radius: 8px;
    }
    .stError {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

PAGES = {
    "Dataset_Selection": Dataset_Selection,
    "Table_Selection": Table_Selection,
    "File_Upload": File_Upload,
    "Prediction": Prediction
}

def main():
    init_state()
    
    # Sidebar navigation (optional)
    with st.sidebar:
        st.title("🧭 Navigation")
        current_page = st.session_state.current_page
        
        # Show current progress
        pages = ["Dataset_Selection", "Table_Selection", "File_Upload", "Prediction"]
        for i, page in enumerate(pages):
            if page == current_page:
                st.write(f"**{i+1}. {page.replace('_', ' ')}** ✅")
            elif i < pages.index(current_page):
                st.write(f"{i+1}. {page.replace('_', ' ')} ✅")
            else:
                st.write(f"{i+1}. {page.replace('_', ' ')}")
    
    # Main content
    page = PAGES.get(st.session_state.current_page, Dataset_Selection)
    page.show()

if __name__ == "__main__":
    main()
