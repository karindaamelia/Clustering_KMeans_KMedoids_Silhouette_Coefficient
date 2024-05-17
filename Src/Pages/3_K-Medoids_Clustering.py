import streamlit as st 

from Utils.Session import get_session_state_instance
from Utils.Clustering import Clustering

# Get the singleton instance for session state
session_state = get_session_state_instance()

st.header("K-Medoids Clustering with Silhouette Coefficent")

# Get the processed dataset from the session state
preprocessing_dataset = session_state.preprocessing_dataset

# Check if the preprocessing dataset is available in 
if preprocessing_dataset is not None:
    st.subheader("Preprocessing Dataset")
    dataset = preprocessing_dataset
    st.dataframe(dataset)
    
    # Integrated Silhouette
    st.subheader("Silhouette Coefficient and Score")
    clustering = Clustering(dataset)
    clustering.apply_clustering(cluster_type='K-Medoids')  
    
else:
    st.warning("Preprocessing dataset is not available. Please go back to the homepage and preprocessing data.")
