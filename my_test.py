import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

st.title('Hello StreamLit')

DATE_COLUMN = 'date/time'




# st.markdown('上传文件')
uploaded_file = st.file_uploader("选择一个文件")


@st.cache_data
def load_data(file_path,nrows=200):
    data = pd.read_csv(file_path, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
# Check if file is uploaded
# Use the 'data' dataframe for further processing and analysis.
if uploaded_file is not None:
    data = load_data(uploaded_file.name)
    st.write(data)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    # data_load_state.text("Done! (using st.cache_data)")
    st.subheader('Raw data')
    st.write(data)
    st.subheader('Number of pickups by hour')
    hist_values = np.histogram(
        data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    st.bar_chart(hist_values)
    st.subheader('Map of all pickups')
    st.map(data)
    hour_to_filter = 17
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
    st.subheader(f'Map of all pickups at {hour_to_filter}:00')
    st.map(filtered_data)
    hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h

    st.subheader('Raw data')
    st.write(data)

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)
else:
    st.write("请上传文件")

