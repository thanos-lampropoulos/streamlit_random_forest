import streamlit as st

st.title("Housing Prices Prediction")

st.header("(my model)")

st.write("""
### Project description
### Using an imputer-MinMaxScaler-PCA_Random_Forest model
We have trained several models to predict the price of a house based on features such as the area of the house and the condition and quality of their different rooms.

""")

import pickle
model = pickle.load(open('thanos-lampropoulos/test_streamlit/main/trained_pipe_random_forest_PCA.sav', 'rb'))

LotFrontage = st.number_input("Lot Frontage")
LotArea = st.number_input("Lot Area")
TotalBsmtSF = st.number_input("Basement Square Feet")
BedroomAbvGr = st.number_input("Number of Bedrooms")
GarageCars = st.number_input("Car spaces in Garage")
GarageArea = st.number_input("Garage area (square feet)")
GarageQual = st.selectbox("Garage quality (select)", ["NA", "TA", "Po", "Fa", "Gd", "Ex"])
Fireplaces = st.number_input("Fireplaces (integer)")


import numpy as np
import pandas as pd

new_house = pd.DataFrame({
    'LotArea':[LotArea],
    'TotalBsmtSF':[TotalBsmtSF],
    'BedroomAbvGr':[BedroomAbvGr],
    'GarageCars':[GarageCars],
    'GarageArea':[GarageArea],

    'YearRemodAdd':[np.NaN], 
    'Heating':[np.NaN], 
    'OverallCond':[np.NaN], 
    'GarageQual':[GarageQual], 
    'KitchenAbvGr':[np.NaN], 
    'LotShape':[np.NaN], 
    'MasVnrArea':[np.NaN], 
    'BsmtUnfSF':[np.NaN], 
    'PoolQC':[np.NaN], 
    'OpenPorchSF':[np.NaN], 
    'Street':[np.NaN], 
    'BsmtQual':[np.NaN], 
    'HalfBath':[np.NaN], 
    'RoofMatl':[np.NaN], 
    'LandSlope':[np.NaN], 
    'PavedDrive':[np.NaN], 
    'OverallQual':[np.NaN], 
    'LowQualFinSF':[np.NaN], 
    'GarageCond':[np.NaN], 
    '3SsnPorch':[np.NaN], 
    'CentralAir':[np.NaN], 
    'GarageYrBlt':[np.NaN], 
    'Utilities':[np.NaN], 
    'Exterior1st':[np.NaN], 
    'Condition1':[np.NaN], 
    'Exterior2nd':[np.NaN], 
    'KitchenQual':[np.NaN], 
    'PoolArea':[np.NaN], 
    'BsmtExposure':[np.NaN], 
    'BsmtFinType2':[np.NaN], 
    'WoodDeckSF':[np.NaN], 
    'MiscVal':[np.NaN], 
    'Fireplaces':[Fireplaces], 
    'HouseStyle':[np.NaN], 
    '1stFlrSF':[np.NaN], 
    'GarageFinish':[np.NaN], 
    'BsmtCond':[np.NaN], 
    'ExterQual':[np.NaN], 
    'Fence':[np.NaN], 
    '2ndFlrSF':[np.NaN], 
    'MSSubClass':[np.NaN], 
    'BsmtFinSF1':[np.NaN], 
    'BsmtFullBath':[np.NaN], 
    'BsmtFinSF2':[np.NaN], 
    'Foundation':[np.NaN], 
    'Electrical':[np.NaN], 
    'BsmtHalfBath':[np.NaN], 
    'FireplaceQu':[np.NaN], 
    'BsmtFinType1':[np.NaN], 
    'RoofStyle':[np.NaN], 
    'LotConfig':[np.NaN], 
    'MasVnrType':[np.NaN], 
    'Functional':[np.NaN], 
    'LotFrontage':[LotFrontage], 
    'GarageType':[np.NaN], 
    'MSZoning':[np.NaN], 
    'ScreenPorch':[np.NaN], 
    'Condition2':[np.NaN], 
    'Neighborhood':[np.NaN], 
    'FullBath':[np.NaN], 
    'BldgType':[np.NaN], 
    'YearBuilt':[np.NaN], 
    'GrLivArea':[np.NaN], 
    'ExterCond':[np.NaN], 
    'TotRmsAbvGrd':[np.NaN], 
    'Alley':[np.NaN], 
    'LandContour':[np.NaN], 
    'HeatingQC':[np.NaN], 
    'EnclosedPorch':[np.NaN], 
    'MiscFeature':[np.NaN]


})

prediction = model.predict(new_house)

st.write("The price of the house is:", prediction)


map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

