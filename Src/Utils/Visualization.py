import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import scipy.stats as stats
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
    
    def histogram_distribution(self):
        start_time = time.time()
        numeric_columns = self.get_numeric_features()
        n_cols = 2  # Jumlah kolom dalam grid
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Jumlah baris dalam grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        
        for i, feature in enumerate(numeric_columns):
            row = i // n_cols
            col = i % n_cols
            sns.histplot(data=self.dataset, x=feature, kde=True, ax=axes[row, col])
            axes[row, col].set_title(f'Histogram for {feature}')
        
        # Menghapus subplot yang tidak terpakai
        for i in range(len(numeric_columns), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        st.pyplot()
        self.computing_time(start_time)
    
    def qq_plot(self):
        start_time = time.time()
        numeric_columns = self.get_numeric_features()
        n_cols = 2  # Jumlah kolom dalam grid
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Jumlah baris dalam grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        
        for i, feature in enumerate(numeric_columns):
            row = i // n_cols
            col = i % n_cols
            stats.probplot(self.dataset[feature], dist="norm", plot=axes[row, col])
            axes[row, col].set_title(f'QQ Plot for {feature}')
        
        # Menghapus subplot yang tidak terpakai
        for i in range(len(numeric_columns), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        st.pyplot()
        self.computing_time(start_time)
        
    # def boxplot(self):
    #     start_time = time.time()
    #     numeric_columns = self.get_numeric_features()
    #     n_cols = 2  # Jumlah kolom dalam grid
    #     n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Jumlah baris dalam grid
    #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        
    #     for i, feature in enumerate(numeric_columns):
    #         row = i // n_cols
    #         col = i % n_cols
    #         sns.boxplot(data=self.dataset, y=feature, ax=axes[row, col])
    #         axes[row, col].set_title(f'Boxplot for {feature}')
        
    #     # Menghapus subplot yang tidak terpakai
    #     for i in range(len(numeric_columns), n_rows * n_cols):
    #         row = i // n_cols
    #         col = i % n_cols
    #         fig.delaxes(axes[row, col])
        
    #     plt.tight_layout()
    #     st.pyplot()
    #     self.computing_time(start_time)
    
    def boxplot(self):
        start_time = time.time()
        numeric_columns = self.get_numeric_features()
        n_plots = len(numeric_columns)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        
        for i, feature in enumerate(numeric_columns):
            row = i // n_cols
            col = i % n_cols
            sns.boxplot(data=self.dataset, x=feature, orient='h', ax=axes[row, col])
            axes[row, col].set_title(f'Boxplot for {feature}')
            axes[row, col].set_ylabel(feature)  # Atur label sumbu y untuk setiap plot
            
        # Menghapus subplot yang tidak terpakai
        for i in range(n_plots, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        st.pyplot()
        self.computing_time(start_time)
