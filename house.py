import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
#Title

col = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup']

st.title('california housing price prediction')

st.image('https://akm-img-a-in.tosshub.com/businesstoday/images/story/202404/660a61cf38b42-the-newly-purchased-house-would-be-subjected-to-a-lock-in-period-of-up-to-3-years-from-the-date-of-p-012709511-16x9.jpg')

st.header('a model of housing prices to predict median house values in California',divider = True)

st.subheader('''User Must Enter Given  Values To Predict Price:
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

st.sidebar.title('Select House Features 🏠')

st.sidebar.image('https://user-images.githubusercontent.com/26305084/117038955-35c4c980-acd6-11eb-9a5e-4e98d4d4b764.gif')

# read_data
temp_df = pd.read_csv('california.csv')

random.seed(12)

all_values = []

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])
    var = st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value), int(max_value)))  

    all_values.append(var)

ss = StandardScaler()
final_value = ss.fit_transform([all_values])

with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

    price = chatgpt.predict(final_value)[0]

    if price>0:
        body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'

        st.success(body)

    else:
        body = 'Invalid House features Values'
        st.warning(body)

