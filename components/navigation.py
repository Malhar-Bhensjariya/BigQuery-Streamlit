import streamlit as st

def init_state():
    """Initialize all session state variables"""
    if 'current_page' not in st.session_state:
        st.session_state.update({
            'current_page': "Dataset_Selection",
            'selected_dataset': None,
            'selected_table': None,
            'mode': None,
            'vertex_model': None
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
    pages = ["Dataset_Selection", "Table_Selection", "File_Upload"]
    current_index = pages.index(st.session_state.current_page)
    return pages[current_index - 1] if current_index > 0 else pages[0]
