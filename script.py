import pandas as pd
import numpy as np
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import plotly.express as px
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import difflib  
from difflib import SequenceMatcher  

import pickle
import os

from sklearn.preprocessing import RobustScaler

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: white;'>Clustering example on Iris Dataset</h1>", unsafe_allow_html=True)

image = Image.open("iris.png")
st.image(image, caption = "Species of Iris flower", width=1970)

st.write("##")
st.write("##")

col1, col2 = st.columns(2)

with col1:
    st.header("About Dataset")
    st.subheader("Context")
    st.write("The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 \
    paper The use of multiple measurements in taxonomic problems. It is sometimes called Anderson's Iris data set because Edgar Anderson collected \
    the data to quantify the morphologic variation of Iris flowers of three related species. The data set consists of 50 samples from each of three \
    species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and the width of \
    the sepals and petals, in centimeters. This dataset became a typical test case for many statistical classification techniques in machine learning \
    such as support vector machines")

    st.subheader("Content")
    st.write("The dataset contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).")

    st.subheader("Acknowledgements")
    st.write("This dataset is free and is publicly available at the UCI Machine Learning Repository")


with col2:
    df = pd.read_csv("Iris.csv")
    st.dataframe(df)

df.drop(["Id"], axis=1, inplace = True)

st.write("##")
st.write("##")

k_means = KMeans(n_clusters=3).fit(df.drop("Species", axis = 1))
hc = AgglomerativeClustering(n_clusters=3).fit_predict(df.drop("Species", axis = 1))

df["KMeans Clustering"] = k_means.labels_
df["Hierarchical Clustering"] = hc

df["KMeans Clustering"].replace({2:"Iris-virginica", 1:"Iris-setosa", 0:"Iris-versicolor"}, inplace = True)
df["Hierarchical Clustering"].replace({2:"Iris-virginica", 1:"Iris-setosa", 0:"Iris-versicolor"}, inplace = True)

col1, col2, col3  = st.columns(3)

with col1:
    check1 = st.checkbox('Original species')
    if check1:
        st.write(df.Species)

with col2:
    check2 = st.checkbox('KMeans Clustering')
    if check2:
        st.write(df["KMeans Clustering"])  
        st.write(difflib.SequenceMatcher(None, df["Species"], df["KMeans Clustering"]).ratio())

with col3:
    check3 = st.checkbox('Hierarchical Clustering')
    if check3:
        st.write(df["Hierarchical Clustering"])
        st.write(difflib.SequenceMatcher(None, df["Species"], df["Hierarchical Clustering"]).ratio())

pca = PCA(n_components = 1)
sepal_pca = pca.fit_transform(X=df[["SepalLengthCm","SepalWidthCm"]])
sepal_pca = pd.DataFrame(data = sepal_pca, columns=["Original Sepal"])

petal_pca = pca.fit_transform(X=df[["PetalLengthCm", "PetalWidthCm"]])
petal_pca = pd.DataFrame(data = petal_pca, columns=["Original Petal"])

df_pca = pd.concat([sepal_pca, petal_pca, df[["Species", "Hierarchical Clustering", "KMeans Clustering"]]], axis = 1)

col1, col2, col3  = st.columns(3)
st.set_option('deprecation.showPyplotGlobalUse', False)


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.scatterplot(x = df_pca["Original Sepal"], y = df_pca["Original Petal"], data = df_pca, hue = "Species")
plt.title("Original Species")

with col1:
    st.pyplot()

plt.subplot(1,3,2)
sns.scatterplot(x = df_pca["Original Sepal"], y = df_pca["Original Petal"], data = df_pca, hue = "KMeans Clustering")
plt.title("Result of KMeans Clustering")

with col2:
    st.pyplot()

plt.subplot(1,3,3)
sns.scatterplot(x = df_pca["Original Sepal"], y = df_pca["Original Petal"], data = df_pca, hue = "Hierarchical Clustering")
plt.title("Result of Hierarchical Clustering")

with col3:
    st.pyplot()

