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
            
            # Display tables with options
            for tbl in tables:
                st.write(f"**{tbl}**")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button(f"Append Data", key=f"append_{tbl}", use_container_width=True):
                        st.session_state.update({
                            'selected_table': tbl,
                            'mode': 'append'
                        })
                        navigate_to("File_Upload")
                
                with col2:
                    if st.button(f"Train or Predict", key=f"predict_{tbl}", use_container_width=True):
                        st.session_state.update({
                            'selected_table': tbl,
                            'mode': 'predict'
                        })
                        navigate_to("Prediction")
                
                st.divider()
                
    except Exception as e:
        st.error(f"Error loading tables: {str(e)}")

    # Create new table
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