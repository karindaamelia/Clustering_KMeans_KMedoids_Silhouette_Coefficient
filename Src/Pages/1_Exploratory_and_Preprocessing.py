import streamlit as st

from Utils.Session import get_session_state_instance
from Utils.Exploratory import Exploratory
from Utils.Visualization import Visualization
from Utils.Preprocessing import Preprocessing

def visualize_data(visualization, dataset):
    visualizer = Visualization(dataset)
    
    if visualization == 'Histogram Distribution':
        visualizer.histogram_distribution()
    elif visualization == 'QQ Plot':
        visualizer.qq_plot()
    elif visualization == 'Box Plot':
        visualizer.boxplot()
    elif visualization == 'None':
        # If "None" is selected, there is no need to visualization
        pass
        
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
    
# Get the singleton instances
session_state = get_session_state_instance()

st.header("Exploratory Data Analysis and Preprocessing Dataset")

# Check if dataset is available and is a DataFrame
if session_state.dataset_uploaded:

    # Get the dataset from data_loader
    dataset = session_state.get_dataset()
    st.dataframe(dataset)
    
    # Reset variabel selected_features di session state
    st.session_state.selected_features = []
    
    # Integrate Exploratory
    data_displayer = Exploratory(dataset)
    st.subheader("Dataset Information")
    data_displayer.basic_info()
    data_displayer.table_info()
    st.subheader("Descriptive Statistics")
    data_displayer.descriptive_statistics()
    
    # Selectbox for Visualization
    st.subheader("Visualization")
    visualization = st.selectbox(
        'Select a Visualization',
        ('None', 'Histogram Distribution', 'QQ Plot', 'Box Plot')
    )
    visualize_data(visualization, dataset)
    
    # Integrate Preprocessing
    if not hasattr(session_state, 'preprocessing'):
        preprocessing = Preprocessing(dataset)
        session_state.preprocessing = preprocessing
    else:
        preprocessing = session_state.preprocessing
        
    preprocessing.reset(dataset)
    preprocessing.perform_preprocessing() 
    preprocessing.label_encode()
    preprocessing.power_transformation()
    preprocessing.select_features()
    preprocessing.rename_attributes()
    preprocessing.display_preprocessing_dataset()
    
    # Save the preprocessing dataset to session state
    session_state.preprocessing_dataset = preprocessing.get_preprocessing_dataset()
    
    # Check if the dataset is successfully saved to session state
    if session_state.preprocessing_dataset is not None:
        st.success("Preprocessing completed and dataset saved to session state.")
    else:
        st.error("Error in saving the preprocessed dataset.")

else:
    # session_state.show_no_dataset_warning()
    st.warning("Please upload a dataset first on the homepage.")
    