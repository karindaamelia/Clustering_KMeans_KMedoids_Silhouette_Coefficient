import streamlit as st

class Config:
    @staticmethod
    def set_page_config():
        st.set_page_config(
            page_title="Clustering Web Application Using K-Means and K-Medoids",
        )
    
    @staticmethod
    def disable_pyplot_warning():
        # Menghapus atau memberikan try-except pada opsi konfigurasi yang tidak valid
        try:
            st.set_option('deprecation.showPyplotGlobalUse', False)
        except Exception as e:
            print(f"Warning: {e}")
    
    @staticmethod
    def set_global_config():
        Config.set_page_config()
        Config.disable_pyplot_warning()
