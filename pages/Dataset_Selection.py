import streamlit as st
from components.gcp_clients import get_clients
from components.navigation import navigate_to, init_state

def show():
    init_state()
    clients = get_clients()
    
    st.title("🗂️ Select BigQuery Dataset")
    
    # List existing datasets
    try:
        datasets = [ds.dataset_id for ds in clients['bq'].list_datasets()]
        if datasets:
            st.subheader("📋 Available Datasets")
            cols = st.columns(2)
            for i, ds in enumerate(datasets):
                with cols[i % 2]:
                    if st.button(ds, key=f"ds_{ds}", use_container_width=True):
                        st.session_state.selected_dataset = ds
                        navigate_to("Table_Selection")
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")

    # Create new dataset
    st.divider()
    st.subheader("➕ Create New Dataset")
    
    with st.form("new_dataset_form"):
        new_ds = st.text_input("Dataset name", placeholder="e.g., my_dataset")
        submit = st.form_submit_button("Create Dataset", use_container_width=True)
        
        if submit and new_ds:
            try:
                clients['bq'].create_dataset(f"{clients['bq'].project}.{new_ds}")
                st.session_state.selected_dataset = new_ds
                st.success(f"Dataset '{new_ds}' created successfully!")
                navigate_to("Table_Selection")
            except Exception as e:
                st.error(f"Creation failed: {str(e)}")
