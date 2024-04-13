import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
import streamlit as st 
import matplotlib.pyplot as plt

import PIL.Image
if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image

#css
page_bg_color = """
<style>
[data-testid="stAppViewContainer"] {
    background-color : #000000;

}
</style>
"""

st.markdown(page_bg_color, unsafe_allow_html=True)

model = load_model('Stock Predictions Model.keras')

text_1 = '<h1 style="font-family:Arial Bold; color:White;">Stock Market Predictor</h1>'
st.markdown(text_1, unsafe_allow_html=True)
# st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2024-02-13'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
text_2 = '<p style="font-family:Arial; color:White;">Stock price daily report.</p>'
st.markdown(text_2, unsafe_allow_html=True)
st.write(data)



data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
text_3 = '<p style="font-family:Arial; font-size:20; color:White;">Current price vs Moving Average last 50 days.</p>'
st.markdown(text_3, unsafe_allow_html=True)
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.gca().spines['bottom'].set_color('white')  # X-axis
plt.gca().spines['left'].set_color('white') 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tick_params(colors='white', which='both')
plt.gcf().set_facecolor('none')
plt.gca().set_facecolor('none')  # RGBA tuple with alpha=0.5
plt.xlabel('Year',color='white')
plt.ylabel('Price', color = 'white')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
text_4 = '<p style="font-family:Arial; font-size:20; color:White;">Current price vs Moving Average last 50 days vs Moving Average last 100 days.</p>'
st.markdown(text_4, unsafe_allow_html=True)
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.gca().spines['bottom'].set_color('white')  # X-axis
plt.gca().spines['left'].set_color('white') 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tick_params(colors='white', which='both')
plt.gcf().set_facecolor('none')
plt.gca().set_facecolor('none')  # RGBA tuple with alpha=0.5
plt.xlabel('Year',color='white')
plt.ylabel('Price', color = 'white')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
text_5 = '<p style="font-family:Arial; font-size:20; color:White;">Current price vs Moving Average last 50 days vs Moving Average last 100 days vs Moving Average last 200 days.</p>'
st.markdown(text_5, unsafe_allow_html=True)
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.gca().spines['bottom'].set_color('white')  # X-axis
plt.gca().spines['left'].set_color('white') 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tick_params(colors='white', which='both')
plt.gcf().set_facecolor('none')
plt.gca().set_facecolor('none')  # RGBA tuple with alpha=0.5
plt.xlabel('Year',color='white')
plt.ylabel('Price', color = 'white')
plt.show()
st.pyplot(fig3)

# Area Slicing
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    # ex: take the first 100 data, to calculate the 101th data 
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale 
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label = 'Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.gca().spines['bottom'].set_color('white')  # X-axis
plt.gca().spines['left'].set_color('white') 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tick_params(colors='white', which='both')
plt.gcf().set_facecolor('none')
plt.gca().set_facecolor('none')  # RGBA tuple with alpha=0.5
plt.xlabel('Time',color='white')
plt.ylabel('Price', color = 'white')
plt.show()
st.pyplot(fig4)