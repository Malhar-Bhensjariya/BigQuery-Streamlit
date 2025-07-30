import streamlit as st
import os
import requests
import time
from dotenv import load_dotenv
from components.navigation import init_state
from pages import (
    Dataset_Selection,
    Table_Selection,
    File_Upload,
    Prediction
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="BigQuery Streamlit",
    page_icon="☁️",
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
    .service-status {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .service-healthy {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .service-unhealthy {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .service-unconfigured {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

PAGES = {
    "Dataset_Selection": Dataset_Selection,
    "Table_Selection": Table_Selection,
    "File_Upload": File_Upload,
    "Prediction": Prediction
}

def check_service_health(url, service_name, timeout=10):
    """Check if a service is healthy and accessible"""
    if not url or url.strip() == "":
        return False, f"{service_name} URL not configured", 0
    
    try:
        start_time = time.time()
        
        # Try different endpoints
        endpoints = ["/health", "/", ""]
        
        for endpoint in endpoints:
            try:
                test_url = url.rstrip('/') + endpoint
                response = requests.get(test_url, timeout=timeout)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    return True, f"{service_name} is ready", response_time
                elif response.status_code == 404 and endpoint != "":
                    continue
                else:
                    return False, f"{service_name} returned status {response.status_code}", response_time
                    
            except requests.exceptions.RequestException:
                if endpoint != "":
                    continue
                else:
                    response_time = time.time() - start_time
                    return False, f"{service_name} is not accessible", response_time
                    
    except Exception as e:
        return False, f"{service_name} check failed: {str(e)}", 0

def get_services_status():
    """Get status of all services"""
    training_url = os.environ.get('TRAINING_URL', '').strip()
    predict_url = os.environ.get('PREDICT_URL', '').strip()
    
    services = {
        'training': {
            'name': 'Training Service',
            'url': training_url,
            'healthy': False,
            'message': '',
            'response_time': 0
        },
        'prediction': {
            'name': 'Prediction Service', 
            'url': predict_url,
            'healthy': False,
            'message': '',
            'response_time': 0
        }
    }
    
    # Check each service
    for key, service in services.items():
        if service['url']:
            healthy, message, response_time = check_service_health(
                service['url'], service['name']
            )
            service.update({
                'healthy': healthy,
                'message': message,
                'response_time': response_time
            })
        else:
            service['message'] = f"{service['name']} URL not configured"
    
    return services

def show_service_status_compact():
    """Show compact service status in sidebar"""
    services = get_services_status()
    
    st.markdown("### 🌐 Service Status")
    
    for key, service in services.items():
        if service['healthy']:
            st.markdown(f"🟢 {service['name']}: Ready")
        elif service['url']:
            st.markdown(f"🔴 {service['name']}: Not Ready")
        else:
            st.markdown(f"🟡 {service['name']}: Not Configured")
    
    if st.button("🔄 Refresh", key="sidebar_refresh"):
        st.rerun()

def show_service_deployment_check():
    """Show initial service deployment check"""
    st.markdown("## 🚀 Welcome to BigQuery Streamlit AutoML")
    
    st.markdown("""
    This platform helps you build and deploy machine learning models using BigQuery ML.
    Let's check if your backend services are properly deployed.
    """)
    
    with st.spinner("Checking service deployment status..."):
        services = get_services_status()
    
    st.markdown("### 🔍 Service Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        service = services['training']
        st.markdown("**🏋️ Training Service**")
        
        if service['healthy']:
            st.success(f"✅ {service['message']}")
            st.caption(f"Response time: {service['response_time']:.2f}s")
        elif service['url']:
            st.error(f"❌ {service['message']}")
            if service['response_time'] > 0:
                st.caption(f"Response time: {service['response_time']:.2f}s")
        else:
            st.warning("⚠️ Training service URL not configured")
        
        if service['url']:
            st.caption(f"URL: {service['url']}")
    
    with col2:
        service = services['prediction']
        st.markdown("**🔮 Prediction Service**")
        
        if service['healthy']:
            st.success(f"✅ {service['message']}")
            st.caption(f"Response time: {service['response_time']:.2f}s")
        elif service['url']:
            st.error(f"❌ {service['message']}")
            if service['response_time'] > 0:
                st.caption(f"Response time: {service['response_time']:.2f}s")
        else:
            st.warning("⚠️ Prediction service URL not configured")
        
        if service['url']:
            st.caption(f"URL: {service['url']}")
    
    # Overall status and actions
    all_healthy = services['training']['healthy'] and services['prediction']['healthy']
    any_configured = services['training']['url'] or services['prediction']['url']
    
    st.markdown("---")
    
    if not any_configured:
        st.error("🚨 **Services Not Deployed**")
        st.markdown("""
        **To deploy your services:**
        1. Open terminal in your project root directory
        2. Run: `cd services && bash deploy.sh`
        3. Wait for deployment to complete
        4. Restart this Streamlit app
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Check Again", type="primary"):
                st.rerun()
        with col2:
            if st.button("📱 Continue Anyway"):
                st.session_state.services_checked = True
                st.rerun()
        
        return False
        
    elif not all_healthy:
        st.warning("⚠️ **Some Services Not Ready**")
        st.markdown("""
        Services may still be starting up. If this persists:
        - Check Google Cloud Console for deployment status
        - Review Cloud Run logs for errors
        - Ensure all services are properly deployed
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Check", type="primary"):
                st.rerun()
        with col2:
            if st.button("📱 Continue Anyway"):
                st.session_state.services_checked = True
                st.rerun()
        
        return False
        
    else:
        st.success("✅ **All Services Ready**")
        st.markdown("Your AutoML pipeline is fully operational!")
        
        if st.button("🚀 Continue to Application", type="primary"):
            st.session_state.services_checked = True
            st.rerun()
        
        return True

def check_services_for_prediction():
    """Check if services are ready for prediction functionality"""
    services = get_services_status()
    
    if not services['prediction']['healthy']:
        st.error("🚨 **Prediction Service Required**")
        st.markdown("""
        The prediction functionality requires the prediction service to be running.
        
        **Current Status:**
        """)
        
        if services['prediction']['url']:
            st.error(f"❌ {services['prediction']['message']}")
        else:
            st.warning("⚠️ Prediction service URL not configured")
        
        st.markdown("""
        **To fix this:**
        1. Ensure services are deployed: `cd services && bash deploy.sh`
        2. Wait for services to start up
        3. Refresh this page
        """)
        
        if st.button("🔄 Check Service Status"):
            st.rerun()
        
        return False
    
    return True

def main():
    init_state()
    
    # Initialize service check state
    if 'services_checked' not in st.session_state:
        st.session_state.services_checked = False
    
    # Show service deployment check on first run
    if not st.session_state.services_checked:
        show_service_deployment_check()
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
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
        
        st.markdown("---")
        
        # Show service status in sidebar
        show_service_status_compact()
        
        st.markdown("---")
        
        # Reset services check button
        if st.button("🔍 Check Services Again"):
            st.session_state.services_checked = False
            st.rerun()
    
    # Main content
    current_page = st.session_state.current_page
    
    # Special handling for prediction page
    if current_page == "Prediction":
        if not check_services_for_prediction():
            return
    
    # Show the current page
    page = PAGES.get(current_page, Dataset_Selection)
    page.show()

if __name__ == "__main__":
    main()