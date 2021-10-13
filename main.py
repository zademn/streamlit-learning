import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler

import time
import streamlit as st
from streamlit.state.session_state import SessionState
import matplotlib.pyplot as plt

# Add caching on long computations


@st.experimental_memo
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data['data'], data['target']], axis=1)
    df = df.dropna()
    time.sleep(1)
    st.session_state.df = df
    return df


@st.experimental_memo(max_entries=5)
def get_hist(df, col_name):
    return np.histogram(df[col_name], bins=24)[0]


st.title("Predicting house pricing")
col1, col2 = st.columns(2)
if "df" not in st.session_state:
    data_load_state = col1.text("Loading data")  # Set text object
    load_data()
    #st.session_state.df = load_data()
    data_load_state.text("Data loaded")  # Change the text
else:
    data_load_state = col1.text("Data loaded")

df = st.session_state.df


# Sidebar
st.sidebar.subheader("Page navigation")
page = st.sidebar.selectbox(label="Label", options=[
    "Inspect data", "Regression", "Unsupervised"])

if page == "Inspect data":
    # Add a checkbox to choose when to show data
    if st.checkbox("Show raw data"):
        st.subheader("Data")
        st.dataframe(df)

    st.header("Plots")
    st.subheader("Histogram")
    option = st.selectbox(label="Feature", options=list(df.columns))
    hist = get_hist(df, option)
    st.bar_chart(hist)

    with st.form("Form name"):
        st.subheader("Scatter plot 2 features against each other")
        col1, col2, col3 = st.columns(3)
        feature1 = col1.selectbox(label="Feature 1", options=list(df.columns))
        feature2 = col2.selectbox(label="Feature 2", options=list(df.columns))
        feature3 = col3.selectbox(
            label="Color feature", options=list(df.columns), index=len(df.columns)-1)
        submitted = st.form_submit_button("Submit")
        if submitted:
            fig = plt.figure(figsize=(10, 10))
            sc = plt.scatter(df[feature1], df[feature2], c=df[feature3])
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.colorbar(sc)
            st.pyplot(fig)


elif page == "Regression":
    test_size = st.slider(label="Select % test size", min_value=0.1, max_value=0.9,
                          help="The percent of data that should be assigned to the testing split")
    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]

    # Button that splits data according to selected split
    # Sets `is_split` to true to show additional stuff
    if st.button(label="Split, train and evaluate"):
        st.write("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size)

        st.write("Training model data...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.success("Model trained!")

        st.write("Evaluating on training and test data...")
        preds_train = model.predict(X_train)

        col1, col2 = st.columns(2)
        col1.metric("MSE on training",
                    value=round(mean_squared_error(y_train, preds_train), 4))
        preds_test = model.predict(X_test)
        col2.metric("MSE on test", value=round(
            mean_squared_error(y_test, preds_test), 4))

        with st.expander("See regression plots"):
            col1, col2 = st.columns(2)
            fig = plt.figure(figsize=(10, 10))
            plt.title("Training predictions")
            plt.scatter(np.arange(len(y_train)), y_train, c='blue')
            plt.scatter(np.arange(len(preds_train)), preds_train, c='orange')
            col1.pyplot(fig)

            fig = plt.figure(figsize=(10, 10))
            plt.title("Training predictions")
            plt.scatter(np.arange(len(y_test)), y_test, c='blue')
            plt.scatter(np.arange(len(preds_test)), preds_test, c='orange')
            col2.pyplot(fig)

elif page == "Unsupervised":
    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]

    st.header("Dimensionality reduction")
    st.write("Show data in 2 dimensions")
    model_option = st.selectbox(label="Select model", options=["SVD", "PCA"])
    if model_option == "SVD":
        model = TruncatedSVD(n_components=2)
        emb = model.fit_transform(X)
    elif model_option == "PCA":
        model = PCA(n_components=2)
        emb = model.fit_transform(X)
    else:
        st.error("Please select one of the available models")

    fig = plt.figure(figsize=(10, 10))
    sc = plt.scatter(*emb.T, c=y)
    plt.colorbar(sc)
    st.pyplot(fig)

else:
    st.write("Page not available yet")
