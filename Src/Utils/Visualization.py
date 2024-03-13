import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import time

class Visualization:
    def __init__(self, dataset):
        self.dataset = dataset
        self.numeric_features = self.get_numeric_features()
    
    def computing_time(self, start_time):
        end_time = time.time()
        compute_time = end_time - start_time
        st.write(f"Computing Time: {compute_time} seconds")
    
    def get_numeric_features(self):
        return [col for col in self.dataset.columns if col != 'ID' and pd.api.types.is_numeric_dtype(self.dataset[col])]
    
    def calculate_figsize(self, num_attributes):
        if num_attributes <= 5:
            return (12, 10)
        elif num_attributes <= 10:
            return (16, 12)
        else:
            return (18, 14)
        
    def pairplot(self):
        start_time = time.time()
        numeric_columns = self.get_numeric_features()

        figsize = self.calculate_figsize(len(numeric_columns))
        pairplot = sns.pairplot(data=self.dataset[numeric_columns], height=3, aspect=1.2)
        pairplot.fig.set_size_inches(figsize)
        st.pyplot()
        self.computing_time(start_time)
        
    def correlation_map(self, subset_attrs=None, annot_size=8, cmap='coolwarm'):
        start_time = time.time()

        # Get correlation matrix for numeric attributes excluding 'ID'
        numeric_columns = self.get_numeric_features()
        numeric_columns_exclude_id = [col for col in numeric_columns if col != 'ID']
        corr_matrix = self.dataset[numeric_columns_exclude_id].corr()

        num_attributes = len(numeric_columns_exclude_id)
        figsize = self.calculate_figsize(num_attributes)

        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, linewidths=.5, annot_kws={"size": annot_size})
        st.pyplot()

        self.computing_time(start_time)