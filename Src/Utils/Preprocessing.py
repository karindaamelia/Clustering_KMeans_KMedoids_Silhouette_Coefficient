import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

class Preprocessing:
    def __init__(self, dataset):
        self.dataset = dataset
        self.all_features = list(self.dataset.columns)
        self.selected_features = None
        self.preprocessing_dataset = None
    
    def set_selected_features(self, selected_features):
        self.selected_features = selected_features

    def get_selected_features(self):
        return self.selected_features
    
    def reset(self, dataset):
        self.dataset = dataset
        self.all_features = list(self.dataset.columns)
        self.selected_features = self.all_features.copy()
        self.preprocessing_dataset = None
        
    def rename_attributes(self):
        renamed_columns = [f"A{i+1}" for i in range(len(self.preprocessing_dataset.columns))]
        self.preprocessing_dataset.columns = renamed_columns
        
    def select_features(self):
        st.subheader("Feature Selection")
        # Set the default value to all features
        selected_features = st.multiselect('Select Features:', self.all_features, default=self.all_features)
        st.write(f"Selected: {selected_features}")
        self.selected_features = selected_features
        
        print("Selected Features:", self.selected_features)
        
        # Remove unselected columns if the dataset has been preprocessed
        if self.preprocessing_dataset is not None:
            self.preprocessing_dataset = self.preprocessing_dataset[self.selected_features]
            
    def drop_column(self, column_name):
        if column_name in self.preprocessing_dataset.columns:
            self.preprocessing_dataset.drop(columns=[column_name], axis=1, inplace=True)
            
    def label_encode(self):
        label_encoder = LabelEncoder()
        
        # Create a copy of the dataset to avoid direct changes to the original dataset
        encoded_dataset = self.dataset.copy()
        
        # Iterate through each column in the dataset
        for column in encoded_dataset.columns:
            column_type = encoded_dataset[column].dtype
            
            # Perform label encoding only for columns with object, boolean, and datetime data types
            if pd.api.types.is_object_dtype(column_type):
                encoded_dataset[column] = label_encoder.fit_transform(encoded_dataset[column])
            if pd.api.types.is_bool_dtype(column_type):
                encoded_dataset[column] = label_encoder.fit_transform(encoded_dataset[column])
            if pd.api.types.is_datetime64_any_dtype(column_type):
                if any(pd.notna(encoded_dataset[column].dt.day)):
                    encoded_dataset[column + '_day'] = encoded_dataset[column].dt.day
                if any(pd.notna(encoded_dataset[column].dt.month)):
                    encoded_dataset[column + '_month'] = encoded_dataset[column].dt.month
                if any(pd.notna(encoded_dataset[column].dt.year)):
                    encoded_dataset[column + '_year'] = encoded_dataset[column].dt.year
                if any(pd.notna(encoded_dataset[column].dt.hour)):
                    encoded_dataset[column + '_hour'] = encoded_dataset[column].dt.hour
                if any(pd.notna(encoded_dataset[column].dt.minute)):
                    encoded_dataset[column + '_minute'] = encoded_dataset[column].dt.minute
                if any(pd.notna(encoded_dataset[column].dt.second)):
                    encoded_dataset[column + '_second'] = encoded_dataset[column].dt.second
        
        # Update the preprocessing dataset
        self.preprocessing_dataset = encoded_dataset
            
    def power_transformation(self, exponent=2):
        if self.preprocessing_dataset is not None and not self.preprocessing_dataset.empty:
            transformed_data = self.preprocessing_dataset[self.selected_features].apply(lambda x: np.power(x, exponent))  
            self.preprocessing_dataset[self.selected_features] = transformed_data
    
    def handle_outliers(self, method='IQR'):
        if self.preprocessing_dataset is not None and not self.preprocessing_dataset.empty:
            if method == 'IQR':
                Q1 = self.preprocessing_dataset[self.selected_features].quantile(0.25)
                Q3 = self.preprocessing_dataset[self.selected_features].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.preprocessing_dataset[self.selected_features] = self.preprocessing_dataset[self.selected_features][~((self.preprocessing_dataset[self.selected_features] < lower_bound) | (self.preprocessing_dataset[self.selected_features] > upper_bound)).any(axis=1)]
        
    def perform_preprocessing(self):
        if not self.selected_features:
            # If none features are selected, return the original dataset
            self.preprocessing_dataset = self.dataset.copy()
        else:
            # Make a copy of the dataset to be processed
            preprocessing_dataset = self.dataset[self.selected_features].copy()
            
            # Perform label encoding and hadle outlier
            self.label_encode()
            self.handle_outliers()

            # Update the processed dataset
            self.preprocessing_dataset = preprocessing_dataset
            
    def display_preprocessing_dataset(self):
        st.subheader("Data Preprocessing")
        # Check if any features are selected
        if not self.selected_features:
            # Show initial dataset with all attributes
            st.dataframe(self.dataset)
        elif self.preprocessing_dataset is not None and not self.preprocessing_dataset.empty and len(self.preprocessing_dataset.columns) > 0:
            # Check if the result DataFrame has columns
            st.dataframe(self.preprocessing_dataset)
        else:
            st.warning("preprocessing dataset is empty or does not have columns. Please select some features.")
            
    def get_preprocessing_dataset(self):
        return self.preprocessing_dataset