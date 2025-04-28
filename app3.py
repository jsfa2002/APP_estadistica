# -*- coding: utf-8 -*-
"""
Aplicación de Análisis Multivariado con Streamlit - Versión con Coeficientes Beta
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno  # ¡IMPORTANTE! necesario para la matriz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report, 
                           mean_squared_error, r2_score, precision_score, recall_score, f1_score, roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from prince import MCA
from collections import defaultdict
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="ReDim - Análisis Multivariado", layout="wide")
st.title('📊 ReDim: Análisis Multivariado Completo')

# CSS personalizado
st.markdown("""
    <style>
    .stApp { background-color: #f5f5f5; }
    h1 { color: #4CAF50; text-align: center; }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
    .model-card { border-radius: 10px; padding: 15px; margin: 10px 0; 
                border-left: 5px solid #4CAF50; background-color: white; }
    .coef-table { margin-top: 20px; border-collapse: collapse; width: 100%; }
    .coef-table th { background-color: #4CAF50; color: white; padding: 8px; text-align: left; }
    .coef-table td { padding: 8px; border-bottom: 1px solid #ddd; }
    .coef-table tr:nth-child(even) { background-color: #f2f2f2; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar para subir el archivo
st.sidebar.header("Opciones de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Archivo cargado exitosamente!")
    
    # Identificar tipos de columnas
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Mostrar vista previa de datos
    st.subheader("📌 Vista previa de los datos")
    st.dataframe(df.head())
    
    # Mostrar información básica del dataset
    st.subheader("🔍 Información del Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Variables Numéricas:**", list(numeric_cols))
    with col2:
        st.write("**Variables Categóricas:**", list(cat_cols))
    
    # Inicializar session_state para almacenar modelos
    if 'model_results' not in st.session_state:
        st.session_state.model_results = defaultdict(dict)
        st.session_state.last_model = None
    
    # Menú de análisis
    analysis_type = st.sidebar.radio("Selecciona el tipo de análisis", 
                                   ["EDA", "Modelos Predictivos", "PCA", "MCA"])

    # ====================== EDA ======================
    if analysis_type == "EDA":
        st.subheader("📊 Análisis Exploratorio de Datos (EDA)")
        
        # Estadísticas descriptivas
        st.write("### Estadísticas descriptivas")
        st.write(df.describe())
        
        # Valores nulos
        st.write("### Valores nulos por columna")
        null_data = df.isnull().sum().reset_index()
        null_data.columns = ['Variable', 'Conteo Nulos']
        st.bar_chart(null_data.set_index('Variable'))

        # Matriz de valores nulos (Nueva parte)
        st.write("### 🔍 Matriz de patrones de valores nulos")
        fig, ax = plt.subplots(figsize=(12, 6))
        msno.matrix(df, ax=ax)
        st.pyplot(fig)
        
        # Distribución de variables numéricas
        if len(numeric_cols) > 0:
            st.write("### Distribución de variables numéricas")
            selected_num = st.selectbox("Selecciona variable numérica", numeric_cols)
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df[selected_num], kde=True, ax=ax[0])
            sns.boxplot(x=df[selected_num], ax=ax[1])
            st.pyplot(fig)
        
        # Conteo de categorías
        if len(cat_cols) > 0:
            st.write("### Conteo de categorías")
            selected_cat = st.selectbox("Selecciona variable categórica", cat_cols)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(y=selected_cat, data=df, ax=ax, order=df[selected_cat].value_counts().index)
            st.pyplot(fig)
        
        # Correlación numérica
        if len(numeric_cols) > 1:
            st.write("### Matriz de correlación")
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    # ====================== MODELOS PREDICTIVOS ======================
    elif analysis_type == "Modelos Predictivos":
        st.subheader("🔮 Modelos Predictivos")

        # Pestañas para organización
        tab1, tab2, tab3 = st.tabs(["🏋️ Entrenar Modelo", "🔍 Comparar Modelos", "📊 Comparar Todos"])

        with tab1:
            # Aqui vendría tu código de entrenar modelos (ya lo tienes)
            pass

        with tab2:
            st.subheader("🔍 Comparar Modelos Individuales")
            # Aqui vendría tu código de comparación de modelos (ya lo tienes)
            pass

        with tab3:
            st.subheader("📊 Comparación Automática de Todos los Modelos")
            # Aqui vendría tu código de comparación global (ya lo tienes)
            pass

    # ====================== PCA ======================
    elif analysis_type == "PCA":
        st.subheader("🔎 Análisis de Componentes Principales (PCA)")
        # Tu código PCA ya existente
        pass

    # ====================== MCA ======================
    elif analysis_type == "MCA":
        st.subheader("🌝 Análisis de Correspondencias Múltiples (MCA)")
        # Tu código MCA ya existente
        pass

st.sidebar.markdown("---")
st.sidebar.markdown("🚀 Desarrollado con cariño por ReDim Team")

