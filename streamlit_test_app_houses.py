import streamlit as st

st.title("Housing Prices Prediction")

st.header("(my model)")

st.write("""
### Project description
### Using an imputer-MinMaxScaler-PCA_Random_Forest model
We have trained several models to predict the price of a house based on features such as the area of the house and the condition and quality of their different rooms.

""")

import pickle
model = pickle.load(open('trained_pipe_random_forest_PCA.sav', 'rb'))

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

    'YearRemodAdd':[np.nan], 
    'Heating':[np.nan], 
    'OverallCond':[np.nan], 
    'GarageQual':[GarageQual], 
    'KitchenAbvGr':[np.nan], 
    'LotShape':[np.nan], 
    'MasVnrArea':[np.nan], 
    'BsmtUnfSF':[np.nan], 
    'PoolQC':[np.nan], 
    'OpenPorchSF':[np.nan], 
    'Street':[np.nan], 
    'BsmtQual':[np.nan], 
    'HalfBath':[np.nan], 
    'RoofMatl':[np.nan], 
    'LandSlope':[np.nan], 
    'PavedDrive':[np.nan], 
    'OverallQual':[np.nan], 
    'LowQualFinSF':[np.nan], 
    'GarageCond':[np.nan], 
    '3SsnPorch':[np.nan], 
    'CentralAir':[np.nan], 
    'GarageYrBlt':[np.nan], 
    'Utilities':[np.nan], 
    'Exterior1st':[np.nan], 
    'Condition1':[np.nan], 
    'Exterior2nd':[np.nan], 
    'KitchenQual':[np.nan], 
    'PoolArea':[np.nan], 
    'BsmtExposure':[np.nan], 
    'BsmtFinType2':[np.nan], 
    'WoodDeckSF':[np.nan], 
    'MiscVal':[np.nan], 
    'Fireplaces':[Fireplaces], 
    'HouseStyle':[np.nan], 
    '1stFlrSF':[np.nan], 
    'GarageFinish':[np.nan], 
    'BsmtCond':[np.nan], 
    'ExterQual':[np.nan], 
    'Fence':[np.nan], 
    '2ndFlrSF':[np.nan], 
    'MSSubClass':[np.nan], 
    'BsmtFinSF1':[np.nan], 
    'BsmtFullBath':[np.nan], 
    'BsmtFinSF2':[np.nan], 
    'Foundation':[np.nan], 
    'Electrical':[np.nan], 
    'BsmtHalfBath':[np.nan], 
    'FireplaceQu':[np.nan], 
    'BsmtFinType1':[np.nan], 
    'RoofStyle':[np.nan], 
    'LotConfig':[np.nan], 
    'MasVnrType':[np.nan], 
    'Functional':[np.nan], 
    'LotFrontage':[LotFrontage], 
    'GarageType':[np.nan], 
    'MSZoning':[np.nan], 
    'ScreenPorch':[np.nan], 
    'Condition2':[np.nan], 
    'Neighborhood':[np.nan], 
    'FullBath':[np.nan], 
    'BldgType':[np.nan], 
    'YearBuilt':[np.nan], 
    'GrLivArea':[np.nan], 
    'ExterCond':[np.nan], 
    'TotRmsAbvGrd':[np.nan], 
    'Alley':[np.nan], 
    'LandContour':[np.nan], 
    'HeatingQC':[np.nan], 
    'EnclosedPorch':[np.nan], 
    'MiscFeature':[np.nan]


})

prediction = model.predict(new_house)

st.write("The price of the house is:", prediction)


map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

