import streamlit as st

class Config:
    @staticmethod
    def set_page_config():
        st.set_page_config(
            page_title="Clustering Web Application Using K-Means and K-Medoids",
        )
    
    @staticmethod
    def disable_pyplot_warning():
        st.set_option('deprecation.showPyplotGlobalUse', False)
    
    @staticmethod
    def set_global_config():
        Config.set_page_config()
        Config.disable_pyplot_warning()