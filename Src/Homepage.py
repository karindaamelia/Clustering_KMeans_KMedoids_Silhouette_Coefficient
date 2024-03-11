import streamlit as st 
import pandas as pd

from Utils.Config import Config
from Utils.Session import get_session_state_instance

# Access global config
Config.set_global_config()

# Get the singleton instances
session_state = get_session_state_instance()

# Title and Sidebar
st.title('Clustering Web Application')
st.header('Using K-Means and K-Medoids Algorithm with Silhouette Coefficient Optimization')
st.sidebar.success("Select a page above")

# Brief points
st.warning("""
Before uploading, make sure the dataset:
- CSV format
- Clean and no missing values
- Data types processed: int, float, object, boolean, and date-time
""")

# File Uploader for dataset
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # read and show dataset
    if uploaded_file.type == "text/csv":
        dataset = pd.read_csv(uploaded_file, sep=None, engine='python')
    else:
        st.error("Unsupported file type. Please upload a CSV file.")
    
    st.success("File uploaded successfully!")
    st.write("Uploaded data:")
    st.dataframe(dataset)
    
    # Save the uploaded dataset to SessionState
    session_state.set_dataset(dataset)

    # Set the session state to indicate that the dataset has been uploaded
    session_state.dataset_uploaded = True
