import streamlit as st
import pandas as pd
import numpy as np
import py7zr
import tempfile
import os
import pickle as pkl
from sklearn.preprocessing import LabelEncoder

archive_path = 'big_mart_data.7z'
csv_filename = 'big_mart_data_backup.csv'

with tempfile.TemporaryDirectory() as temp_dir:
    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        extracted_files = archive.extract(path=temp_dir)

    print("Extracted files:", os.listdir(temp_dir))

    csv_file_path = os.path.join(temp_dir, csv_filename)

    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File not found: {csv_file_path}")
    
    news_data = pd.read_csv(csv_file_path)

loaded_model = pkl.load(open('regressor.pkl', 'rb'))
# big_mart_data = pd.read_csv('big_mart_data_backup.csv')

def predict_sales(Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year,Outlet_Size,Outlet_Location_Type,Outlet_Type):

    Identifier = list(pd.unique(big_mart_data["Item_Identifier"].values))
    Identifier_val = Identifier.index(Item_Identifier)

    Fat_Content = list(pd.unique(big_mart_data["Item_Fat_Content"].values))
    Fat_Content_val = Fat_Content.index(Item_Fat_Content)

    Type = list(pd.unique(big_mart_data["Item_Type"].values))
    Item_Type_val = Type.index(Item_Type)

    Outlet = list(pd.unique(big_mart_data["Outlet_Identifier"].values))
    Outlet_Identifier_val = Outlet.index(Outlet_Identifier)

    Size = list(pd.unique(big_mart_data["Outlet_Size"].values))
    Outlet_Size_val = Size.index(Outlet_Size)

    Location_Type = list(pd.unique(big_mart_data["Outlet_Location_Type"].values))
    Outlet_Location_Type_val = Location_Type.index(Outlet_Location_Type)

    Out_Type = list(pd.unique(big_mart_data["Outlet_Type"].values))
    Outlet_Type_val = Out_Type.index(Outlet_Type)

    encoder = LabelEncoder()

    new_big_mart = big_mart_data.copy()

    new_big_mart["Item_Identifier"] = encoder.fit_transform(big_mart_data["Item_Identifier"])

    new_big_mart["Item_Fat_Content"] = encoder.fit_transform(big_mart_data["Item_Fat_Content"])

    new_big_mart["Item_Type"] = encoder.fit_transform(big_mart_data["Item_Type"])

    new_big_mart["Outlet_Identifier"] = encoder.fit_transform(big_mart_data["Outlet_Identifier"])

    new_big_mart["Outlet_Size"] = encoder.fit_transform(big_mart_data["Outlet_Size"])

    new_big_mart["Outlet_Location_Type"] = encoder.fit_transform(big_mart_data["Outlet_Location_Type"])

    new_big_mart["Outlet_Type"] = encoder.fit_transform(big_mart_data["Outlet_Type"])

    sales_value = np.array([Identifier_val, Item_Weight, Fat_Content_val, Item_Visibility, Item_Type_val,   Item_MRP, Outlet_Identifier_val, Outlet_Establishment_Year, Outlet_Size_val, Outlet_Location_Type_val, Outlet_Type_val])

    sales_value_reshape = sales_value.reshape(1,-1)

    prediction = loaded_model.predict(sales_value_reshape)
    return prediction[0]

# Streamlit UI
st.title('Big Mart Sales Prediction')

# Getting the input data from user
# columns for input field
col1, col2, col3 =  st.columns(3)

# User input fields
with col1:
    Item_Identifier =  st.selectbox('Select Item Identifier:', pd.unique(big_mart_data['Item_Identifier'].values))

with col2:
    Item_Weight = st.number_input('Enter Item Weight', min_value=0.0, max_value=500.0, step=0.10000)

with col3:
    Item_Fat_Content = st.selectbox('Select Item Fat Content:', pd.unique(big_mart_data['Item_Fat_Content'].values))

with col1:
    Item_Visibility = st.number_input('Enter Item Visibility', min_value=0.0, max_value=500.0, step=0.10000)

with col2:
    Item_Type = st.selectbox('Select Item Type:', pd.unique(big_mart_data['Item_Type'].values))

with col3:
    Item_MRP = st.number_input('Enter Item MRP')

with col1:
    Outlet_Identifier = st.selectbox('Select Outlet Identifier:', pd.unique(big_mart_data['Outlet_Identifier'].values))

with col2:
    Outlet_Establishment_Year = st.selectbox('Select Outlet Establishment Year:', pd.unique(big_mart_data['Outlet_Establishment_Year'].values))

with col3:
    Outlet_Size = st.selectbox('Select Outlet Size:', pd.unique(big_mart_data['Outlet_Size'].values))

with col1:
    Outlet_Location_Type = st.selectbox('Select Outlet Location:', pd.unique(big_mart_data['Outlet_Location_Type'].values))

with col2:
    Outlet_Type = st.selectbox('Select Outlet Type:', pd.unique(big_mart_data['Outlet_Type'].values))

# Predict button
if st.button('Predict Sales'):
    # Make prediction
    prediction = predict_sales(Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type,Item_MRP, Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type)

    st.success(f'Predicted Sales is : {prediction}')

