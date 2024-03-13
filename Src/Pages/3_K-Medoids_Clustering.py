import streamlit as st 

from Utils.Session import get_session_state_instance
from Utils.Clustering import Clustering
from Utils.Visualization import Visualization

def visualize_data(visualization, dataset):
    visualizer = Visualization(dataset)
    
    if visualization == 'Histogram Distribution':
        visualizer.histogram_distribution()
    elif visualization == 'QQ Plot':
        visualizer.qq_plot()
    elif visualization == 'Pairplot':
        visualizer.pairplot()
    elif visualization == 'None':
        # If "None" is selected, there is no need to visualization
        pass

# Get the singleton instance for session state
session_state = get_session_state_instance()

st.header("K-Medoids Clustering with Silhouette Coefficent")

# Mendapatkan dataset yang telah diproses dari session state
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
    
    # Selectbox for Visualization
    st.subheader("Other Visualization")
    visualization = st.selectbox(
        'Select a Visualization',
        ('None', 'Histogram Distribution', 'QQ Plot', 'Pairplot')
    )
    visualize_data(visualization, dataset)
    
else:
    st.warning("Preprocessing dataset is not available. Please go back to the Homepage to preprocess the data.")
