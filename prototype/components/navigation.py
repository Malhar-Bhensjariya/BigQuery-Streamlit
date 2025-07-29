import streamlit as st

def init_state():
    """Initialize all session state variables"""
    if 'current_page' not in st.session_state:
        st.session_state.update({
            'current_page': "Dataset_Selection",
            'selected_dataset': None,
            'selected_table': None,
            'mode': None,
            'selected_model': None  # Updated from vertex_model to selected_model for custom AutoML
        })

def navigate_to(page):
    """Change current page"""
    st.session_state.current_page = page
    st.rerun()

def back_button():
    """Generic back button component"""
    if st.button("← Back", key="back_btn"):
        navigate_to(get_previous_page())

def get_previous_page():
    """Determine previous page based on workflow"""
    current_page = st.session_state.current_page
    
    # Define the navigation flow
    if current_page == "Dataset_Selection":
        return "Dataset_Selection"  # Already at the first page
    
    elif current_page == "Table_Selection":
        return "Dataset_Selection"
    
    elif current_page == "File_Upload":
        return "Table_Selection"
    
    elif current_page == "Prediction":
        return "Table_Selection"  # Both File_Upload and Prediction go back to Table_Selection
    
    else:
        # Fallback to Dataset_Selection for any unknown pages
        return "Dataset_Selection"