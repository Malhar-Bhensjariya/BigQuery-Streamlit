import streamlit as st
from components.gcp_clients import get_clients
from components.navigation import navigate_to, back_button

def show():
    clients = get_clients()
    
    st.title(f"Dataset: `{st.session_state.selected_dataset}`")
    back_button()

    # List existing tables
    try:
        tables = [t.table_id for t in 
                  clients['bq'].list_tables(st.session_state.selected_dataset)]
        
        if tables:
            st.subheader("Available Tables")
            cols = st.columns(2)
            for i, tbl in enumerate(tables):
                with cols[i % 2]:
                    if st.button(f"📄 {tbl}", key=f"tbl_{tbl}", use_container_width=True):
                        st.session_state.update({
                            'selected_table': tbl,
                            'mode': 'append'
                        })
                        navigate_to("File_Upload")
    except Exception as e:
        st.error(f"Error loading tables: {str(e)}")

    # Create new table
    st.divider()
    st.subheader("➕ Create New Table")
    
    with st.form("new_table_form"):
        new_tbl = st.text_input("Table name", placeholder="e.g., my_table")
        submit = st.form_submit_button("Create Table", use_container_width=True)
        
        if submit and new_tbl:
            st.session_state.update({
                'selected_table': new_tbl,
                'mode': 'create'
            })
            navigate_to("File_Upload")
