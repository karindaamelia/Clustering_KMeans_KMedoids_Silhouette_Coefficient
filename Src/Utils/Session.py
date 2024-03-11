import streamlit as st

class SessionState:
    def __init__(self):
        self.dataset = None
        self.dataset_uploaded = False
        self.preprocessing_dataset = None
        
    def get_dataset(self):
        return self.dataset
    
    def set_dataset(self, dataset):
        self.dataset = dataset
    
    def show_no_dataset_warning(self):
        st.warning("Please upload a dataset first on the homepage.")

# Singleton pattern to ensure a single instance is used throughout the application
_session_state_instance = SessionState()

def get_session_state_instance():
    # return _session_state_instance
    global _session_state
    try:
        _session_state
    except NameError:
        _session_state = SessionState()
    return _session_state
