# -*- coding: utf-8 -*-
"""Untitled36.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16JBGa3fPCrlrC_AMKrmUer9ckuXgZwq6
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from prince import MCA

# Configuración de la página
st.set_page_config(page_title="ReDim - Análisis de Datos", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #f5f5f5; }
    h1 { color: #4CAF50; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('📊 ReDim: Exploración de Datos con PCA y MCA')
st.write('Bienvenido a ReDim, una herramienta interactiva para realizar análisis exploratorio de datos (EDA), PCA y MCA.')

# Sidebar para subir el archivo
st.sidebar.header("Opciones")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Archivo cargado exitosamente!")

    st.subheader("📌 Vista previa de los datos")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    st.sidebar.subheader("Análisis disponibles:")
    if st.sidebar.button("Ejecutar EDA"):
        st.subheader("📊 Análisis Exploratorio de Datos (EDA)")

        st.write("### Estadísticas descriptivas")
        st.write(df.describe())

        st.write("### Valores nulos por columna")
        st.bar_chart(df.isnull().sum())

        if len(numeric_cols) > 0:
            st.write("### Distribución de variables numéricas")
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                st.pyplot(fig)

        st.success("EDA completado")

    if st.sidebar.button("Ejecutar PCA"):
        st.subheader("🔎 Análisis de Componentes Principales (PCA)")
        if len(numeric_cols) > 1:
            imputer = SimpleImputer(strategy='mean')
            df_filled = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df[numeric_cols])

            pca = PCA()
            pca.fit(df_scaled)

            st.write("### Valores propios (autovalores) de los componentes principales:")
            st.write(pca.explained_variance_)

            st.write("### Varianza explicada por cada componente:")
            st.bar_chart(pca.explained_variance_ratio_)

            df_pca = pca.transform(df_scaled)

            fig, ax = plt.subplots()
            ax.scatter(df_scaled[:, 0], df_scaled[:, 1], alpha=0.5, color='blue')
            ax.set_title('PCA: PC1 vs PC2')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            st.pyplot(fig)

            st.success("PCA ejecutado correctamente")
        else:
            st.error("Se necesitan al menos 2 variables numéricas para ejecutar PCA")

    if st.sidebar.button("Ejecutar MCA"):
        st.subheader("🎭 Análisis de Correspondencias Múltiples (MCA)")
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 1:
            df_cat = df[cat_cols].astype(str)  # Convertir a string por si acaso

            mca = MCA()
            mca.fit(df_cat)

            eigenvalues = mca.eigenvalues_
            total_inertia = sum(eigenvalues)
            explained_inertia = [eig / total_inertia for eig in eigenvalues]

            st.write("### Proporción de inercia explicada por cada dimensión:")
            st.bar_chart(explained_inertia)

            mca_coords = mca.row_coordinates(df_cat)
            fig, ax = plt.subplots()
            ax.scatter(mca_coords.iloc[:, 0], mca_coords.iloc[:, 1], alpha=0.6, color='red')
            ax.set_title("MCA: Dim1 vs Dim2")
            ax.set_xlabel("Dimensión 1")
            ax.set_ylabel("Dimensión 2")
            st.pyplot(fig)

            st.success("MCA ejecutado correctamente")
        else:
            st.warning("Se necesitan al menos dos columnas categóricas para ejecutar MCA.")

st.sidebar.markdown("---")
st.sidebar.markdown("🚀 Desarrollado con cariño por ReDim Team")