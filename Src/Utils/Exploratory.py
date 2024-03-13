import streamlit as st
import pandas as pd

class Exploratory:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def basic_info(self):
        st.write(f"Number of rows: {self.dataset.shape[0]}")
        st.write(f"Number of columns: {self.dataset.shape[1]}")
        
    def table_info(self):
        info_df = pd.DataFrame({
            'Column': self.dataset.columns,
            'Data Types': self.dataset.dtypes,
            'Count': self.dataset.count(),
            'Null Values': self.dataset.isnull().sum()
        })
        additional_info = pd.DataFrame({
            'Attribute': ['Memory Usage'],
            'Value': [f"{self.dataset.memory_usage(deep=True).sum() / (1024**2):.2f} MB"]
        })
        st.table(info_df)
        st.write(additional_info)
    
    def descriptive_statistics(self):
        descriptive_stats = self.dataset.describe().T
        st.dataframe(descriptive_stats)
    