# -*- coding: utf-8 -*-
"""
ReDim - Análisis Multivariado Completo
Aplicación web para análisis estadístico y modelado predictivo
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report, 
                           mean_squared_error, r2_score, roc_curve, auc, silhouette_score)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from prince import MCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
from streamlit.components.v1 import html

# Configuración de la página
st.set_page_config(page_title="ReDim - Análisis Multivariado", layout="wide")
st.title('📊 ReDim: Análisis Multivariado Completo')

# CSS personalizado
st.markdown("""
    <style>
    .stApp { 
        background-color: #f9f9f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1 { 
        color: #2c3e50;
        text-align: center;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    .sidebar .sidebar-content { 
        background-color: #ecf0f1;
        padding: 15px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stSelectbox, .stMultiselect {
        margin-bottom: 15px;
    }
    .stDataFrame {
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# Footer
def footer():
    html("""
    <div style="position: fixed; bottom: 0; width: 100%; background-color: #2c3e50; color: white; padding: 10px; text-align: center;">
        <p>© 2023 ReDim Analytics | Powered by Streamlit | <a href="#" style="color: #3498db;">Documentación</a> | <a href="#" style="color: #3498db;">Soporte</a></p>
    </div>
    """)

footer()

# Sidebar para subir el archivo
st.sidebar.header("Opciones de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

if uploaded_file is not None:
    try:
        # Validar tamaño del archivo
        if uploaded_file.size > 50 * 1024 * 1024:  # 50MB
            st.error("El archivo es demasiado grande (límite: 50MB)")
            st.stop()
        
        df = load_data(uploaded_file)
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
        
        # Menú de análisis
        analysis_type = st.sidebar.radio("Selecciona el tipo de análisis", 
                                       ["EDA", "Modelos Predictivos", "PCA", "MCA", "Clustering", "Guardar Resultados"])
        
        # ====================== EDA ======================
        if analysis_type == "EDA":
            st.subheader("📊 Análisis Exploratorio de Datos (EDA)")
            
            eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Descriptivo", "Correlaciones", "Análisis Avanzado"])
            
            with eda_tab1:
                # Estadísticas descriptivas mejoradas
                st.write("### Estadísticas descriptivas completas")
                st.dataframe(df.describe(include='all').style.background_gradient(cmap='Blues')
                
                # Análisis de valores nulos mejorado
                st.write("### Análisis de valores faltantes")
                null_percent = df.isnull().mean() * 100
                null_df = pd.DataFrame({
                    'Columnas': null_percent.index,
                    '% Valores nulos': null_percent.values,
                    'Tipo de dato': df.dtypes.values
                })
                st.dataframe(null_df[null_df['% Valores nulos'] > 0].sort_values('% Valores nulos', ascending=False))
                
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
                    fig = px.bar(df[selected_cat].value_counts(), 
                                title=f'Distribución de {selected_cat}')
                    st.plotly_chart(fig)
            
            with eda_tab2:
                # Correlación numérica interactiva
                if len(numeric_cols) > 1:
                    st.write("### Matriz de correlación interactiva")
                    fig = px.imshow(df[numeric_cols].corr(), 
                                 text_auto=True, 
                                 color_continuous_scale='RdBu_r',
                                 zmin=-1, zmax=1,
                                 width=800, height=800)
                    st.plotly_chart(fig)
                    
                    # Heatmap de correlaciones significativas
                    st.write("### Correlaciones significativas (|r| > 0.7)")
                    corr_matrix = df[numeric_cols].corr().abs()
                    high_corr = corr_matrix[(corr_matrix > 0.7) & (corr_matrix < 1.0)]
                    if high_corr.any().any():
                        fig = px.imshow(high_corr.dropna(how='all'))
                        st.plotly_chart(fig)
                    else:
                        st.info("No hay correlaciones fuertes entre variables (|r| > 0.7)")
            
            with eda_tab3:
                # Análisis de multicolinealidad (VIF)
                if len(numeric_cols) > 1:
                    st.write("### Factor de Inflación de Varianza (VIF)")
                    vif_data = pd.DataFrame()
                    vif_data["Variable"] = numeric_cols
                    vif_data["VIF"] = [variance_inflation_factor(df[numeric_cols].values, i) 
                                      for i in range(len(numeric_cols))]
                    st.dataframe(vif_data.sort_values("VIF", ascending=False))
                    
                    st.markdown("""
                    **Interpretación del VIF:**
                    - VIF < 5: Multicolinealidad moderada
                    - VIF >= 5 y < 10: Multicolinealidad alta
                    - VIF >= 10: Multicolinealidad muy alta (debe tratarse)
                    """)
        
        # ====================== MODELOS PREDICTIVOS ======================
        elif analysis_type == "Modelos Predictivos":
            st.subheader("🔮 Modelos Predictivos Avanzados")
            
            model_choice = st.selectbox("Selecciona el modelo", 
                                      ["Regresión Lineal", 
                                       "Regresión Logística", 
                                       "Random Forest",
                                       "SVM",
                                       "LDA", 
                                       "QDA"])
            
            target_var = st.selectbox("Selecciona la variable dependiente (Y)", df.columns)
            predictor_vars = st.multiselect("Selecciona las variables predictoras (X)", 
                                           df.columns.drop(target_var))
            
            # Nuevas opciones de configuración
            advanced_options = st.expander("Opciones avanzadas")
            with advanced_options:
                test_size = st.slider("Tamaño del conjunto de prueba", 0.1, 0.5, 0.3)
                random_state = st.number_input("Semilla aleatoria", value=42)
                scale_data = st.checkbox("Estandarizar variables", value=True)
            
            if st.button("Ejecutar Modelo"):
                st.subheader(f"📈 Resultados del Modelo: {model_choice}")
                
                X = df[predictor_vars]
                y = df[target_var]
                
                # Manejo de variables categóricas mejorado
                if X.select_dtypes(include=['object']).any().any():
                    X = pd.get_dummies(X, drop_first=True)
                
                # Estandarización
                if scale_data and len(numeric_cols) > 0:
                    scaler = StandardScaler()
                    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
                
                # Partición del dataset
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state)
                
                # Selección del modelo ampliada
                if model_choice == "Regresión Lineal":
                    model = LinearRegression()
                    model_type = "regression"
                elif model_choice == "Regresión Logística":
                    model = LogisticRegression(max_iter=1000)
                    model_type = "classification"
                elif model_choice == "Random Forest":
                    if y.nunique() > 5:  # Heurística para decidir si es regresión o clasificación
                        model = RandomForestRegressor()
                        model_type = "regression"
                    else:
                        model = RandomForestClassifier()
                        model_type = "classification"
                elif model_choice == "SVM":
                    if y.nunique() > 5:
                        model = SVR()
                        model_type = "regression"
                    else:
                        model = SVC(probability=True)
                        model_type = "classification"
                elif model_choice == "LDA":
                    model = LinearDiscriminantAnalysis()
                    model_type = "classification"
                else:
                    model = QuadraticDiscriminantAnalysis()
                    model_type = "classification"
                
                # Entrenamiento y predicción
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Resultados mejorados
                if model_type == "regression":
                    st.write("### Métricas de Regresión")
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"- Error Cuadrático Medio (MSE): {mse:.4f}")
                    st.write(f"- Coeficiente de Determinación (R²): {r2:.4f}")
                    
                    # Gráfico de valores reales vs predichos
                    fig = px.scatter(x=y_test, y=y_pred, 
                                   labels={'x': 'Valores Reales', 'y': 'Valores Predichos'},
                                   title='Valores Reales vs Predichos')
                    fig.add_shape(type='line', line=dict(dash='dash'),
                                x0=y.min(), x1=y.max(),
                                y0=y.min(), y1=y.max())
                    st.plotly_chart(fig)

                    # Mostrar coeficientes beta para modelos lineales
                    if hasattr(model, 'coef_'):
                        st.write("### Coeficientes Beta")
                        coef_df = pd.DataFrame({
                            "Variable": X.columns,
                            "Coeficiente (β)": model.coef_
                        })
                        st.dataframe(coef_df.sort_values("Coeficiente (β)", ascending=False))

                else:
                    st.write("### Matriz de Confusión")
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(cm, text_auto=True, 
                                  labels=dict(x="Predicho", y="Real"),
                                  title="Matriz de Confusión")
                    st.plotly_chart(fig)
                    
                    st.write("### Reporte de Clasificación")
                    st.code(classification_report(y_test, y_pred))
                    
                    # Curva ROC para clasificación binaria
                    if y.nunique() == 2 and hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        fig = px.area(
                            x=fpr, y=tpr,
                            title=f'Curva ROC (AUC = {roc_auc:.2f})',
                            labels=dict(x='Tasa de Falsos Positivos', y='Tasa de Verdaderos Positivos')
                        )
                        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                        st.plotly_chart(fig)
                
                # Importancia de variables para modelos que lo soportan
                if hasattr(model, 'feature_importances_'):
                    st.write("### Importancia de Variables")
                    feat_importances = pd.DataFrame({
                        'Variable': X.columns,
                        'Importancia': model.feature_importances_
                    }).sort_values('Importancia', ascending=False)
                    
                    fig = px.bar(feat_importances, x='Importancia', y='Variable', 
                                orientation='h', title='Importancia de Variables')
                    st.plotly_chart(fig)
                
                st.success(f"{model_choice} ejecutado correctamente")
            
            # Comparación de modelos (solo para clasificación)
            if len(df[target_var].unique()) <= 5:  # Si parece variable categórica
                if st.button("Comparar Modelos de Clasificación"):
                    st.subheader("📊 Comparación de Modelos de Clasificación")
                    
                    X = df[predictor_vars]
                    y = df[target_var]
                    
                    if X.select_dtypes(include=['object']).any().any():
                        X = pd.get_dummies(X, drop_first=True)
                    
                    if scale_data and len(numeric_cols) > 0:
                        scaler = StandardScaler()
                        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state)
                    
                    models = {
                        "Regresión Logística": LogisticRegression(max_iter=1000),
                        "LDA": LinearDiscriminantAnalysis(),
                        "QDA": QuadraticDiscriminantAnalysis(),
                        "Random Forest": RandomForestClassifier(),
                        "SVM": SVC(probability=True)
                    }
                    
                    results = []
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        results.append({"Modelo": name, "Exactitud": acc})
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df.sort_values("Exactitud", ascending=False))
                    
                    fig = px.bar(results_df.sort_values("Exactitud"), 
                               x='Exactitud', y='Modelo', 
                               orientation='h', title='Comparación de Modelos')
                    fig.update_xaxes(range=[0, 1])
                    st.plotly_chart(fig)
                    
                    best_model = results_df.loc[results_df['Exactitud'].idxmax()]
                    st.success(f"Modelo más exacto: {best_model['Modelo']} con {best_model['Exactitud']:.2%}")
        
        # ====================== PCA ======================
        elif analysis_type == "PCA":
            st.subheader("🔎 Análisis de Componentes Principales (PCA)")
            
            if len(numeric_cols) < 2:
                st.error("Se necesitan al menos 2 variables numéricas para ejecutar PCA")
            else:
                # Selección de variables
                selected_vars = st.multiselect("Selecciona variables para PCA", 
                                             numeric_cols, default=list(numeric_cols))
                
                # Opciones de PCA
                n_components = st.slider("Número de componentes", 2, min(10, len(selected_vars)), 2)
                
                if st.button("Ejecutar PCA"):
                    # Preprocesamiento
                    df_pca = df[selected_vars].copy()
                    imputer = SimpleImputer(strategy='mean')
                    df_filled = pd.DataFrame(imputer.fit_transform(df_pca), columns=selected_vars)
                    
                    scaler = StandardScaler()
                    df_scaled = scaler.fit_transform(df_filled)
                    
                    # Aplicar PCA
                    pca = PCA(n_components=n_components)
                    pca.fit(df_scaled)
                    components = pca.transform(df_scaled)
                    
                    # Resultados
                    st.write("### Varianza explicada por cada componente")
                    var_exp = pca.explained_variance_ratio_
                    cum_var_exp = np.cumsum(var_exp)
                    
                    fig = px.bar(x=range(1, n_components+1), y=var_exp,
                                labels={'x': 'Componente principal', 'y': 'Varianza explicada'},
                                title='Varianza explicada por componente')
                    st.plotly_chart(fig)
                    
                    fig = px.line(x=range(1, n_components+1), y=cum_var_exp,
                                labels={'x': 'Componente principal', 'y': 'Varianza explicada acumulada'},
                                title='Varianza acumulada')
                    fig.update_traces(mode='lines+markers')
                    st.plotly_chart(fig)
                    
                    # Gráfico de componentes
                    st.write("### Gráfico de los dos primeros componentes")
                    if n_components >= 2:
                        fig = px.scatter(x=components[:, 0], y=components[:, 1],
                                       labels={'x': f'PC1 ({var_exp[0]*100:.1f}%)',
                                              'y': f'PC2 ({var_exp[1]*100:.1f}%)'},
                                       title='PCA: Componente 1 vs Componente 2')
                        
                        # Si hay una variable categórica, usarla para colorear
                        if len(cat_cols) > 0:
                            color_var = st.selectbox("Variable para colorear puntos", cat_cols)
                            fig = px.scatter(x=components[:, 0], y=components[:, 1],
                                           color=df[color_var],
                                           labels={'x': f'PC1 ({var_exp[0]*100:.1f}%)',
                                                  'y': f'PC2 ({var_exp[1]*100:.1f}%)'},
                                           title=f'PCA coloreado por {color_var}')
                        st.plotly_chart(fig)
                    
                    # Cargas factoriales
                    st.write("### Cargas factoriales (Componentes principales)")
                    loadings = pd.DataFrame(
                        pca.components_.T,
                        columns=[f'PC{i+1}' for i in range(n_components)],
                        index=selected_vars
                    )
                    st.dataframe(loadings.style.background_gradient(cmap='RdBu_r', axis=None))
                    
                    st.success("PCA ejecutado correctamente")
        
        # ====================== MCA ======================
        elif analysis_type == "MCA":
            st.subheader("🎭 Análisis de Correspondencias Múltiples (MCA)")
            
            if len(cat_cols) < 2:
                st.warning("Se necesitan al menos dos columnas categóricas para ejecutar MCA")
            else:
                # Selección de variables
                selected_cats = st.multiselect("Selecciona variables categóricas para MCA", 
                                             cat_cols, default=list(cat_cols)[:min(5, len(cat_cols))])
                
                n_components = st.slider("Número de dimensiones", 2, min(5, len(selected_cats)), 2)
                
                if st.button("Ejecutar MCA"):
                    df_mca = df[selected_cats].copy().astype(str)
                    
                    # Aplicar MCA
                    mca = MCA(n_components=n_components, n_iter=10, random_state=42)
                    mca.fit(df_mca)
                    
                    # Coordenadas
                    mca_coords = mca.row_coordinates(df_mca)
                    mca_col_coords = mca.column_coordinates(df_mca)
                    
                    # Inercia
                    eigenvalues = mca.eigenvalues_
                    total_inertia = sum(eigenvalues)
                    explained_inertia = [eig / total_inertia for eig in eigenvalues]
                    
                    # Resultados
                    st.write("### Inercia explicada por cada dimensión")
                    fig = px.bar(x=range(1, n_components+1), y=explained_inertia,
                                labels={'x': 'Dimensión', 'y': 'Proporción de inercia explicada'},
                                title='Inercia explicada por dimensión')
                    st.plotly_chart(fig)
                    
                    # Gráfico de individuos
                    st.write("### Gráfico de Individuos (dos primeras dimensiones)")
                    fig = px.scatter(x=mca_coords.iloc[:, 0], y=mca_coords.iloc[:, 1],
                                   labels={'x': f"Dimensión 1 ({explained_inertia[0]*100:.1f}%)",
                                          'y': f"Dimensión 2 ({explained_inertia[1]*100:.1f}%)"},
                                   title="MCA: Individuos")
                    fig.add_hline(y=0, line_dash="dot")
                    fig.add_vline(x=0, line_dash="dot")
                    st.plotly_chart(fig)
                    
                    # Gráfico de categorías
                    st.write("### Gráfico de Categorías (dos primeras dimensiones)")
                    mca_col_coords['Variable'] = mca_col_coords.index.str.split('_').str[0]
                    
                    fig = px.scatter(mca_col_coords, x=0, y=1, color='Variable',
                                   labels={'0': f"Dimensión 1 ({explained_inertia[0]*100:.1f}%)",
                                          '1': f"Dimensión 2 ({explained_inertia[1]*100:.1f}%)"},
                                   title="MCA: Categorías")
                    fig.add_hline(y=0, line_dash="dot")
                    fig.add_vline(x=0, line_dash="dot")
                    st.plotly_chart(fig)
                    
                    st.success("MCA ejecutado correctamente")
        
        # ====================== Clustering ======================
        elif analysis_type == "Clustering":
            st.subheader("🔍 Análisis de Clustering")
            
            if len(numeric_cols) < 2:
                st.warning("Se necesitan al menos 2 variables numéricas para clustering")
            else:
                # Selección de variables
                selected_vars = st.multiselect("Selecciona variables para clustering", 
                                             numeric_cols, default=list(numeric_cols)[:5])
                
                # Preprocesamiento
                df_cluster = df[selected_vars].copy()
                imputer = SimpleImputer(strategy='mean')
                df_filled = pd.DataFrame(imputer.fit_transform(df_cluster), columns=selected_vars)
                scaler = StandardScaler()
                df_scaled = scaler.fit_transform(df_filled)
                
                # Método de clustering
                cluster_method = st.selectbox("Método de clustering", ["K-Means", "Jerárquico"])
                
                if cluster_method == "K-Means":
                    # Determinación óptima de clusters
                    st.write("### Método del Codo para determinar K óptimo")
                    inertia = []
                    max_clusters = min(10, len(df_scaled)-1)
                    possible_k = range(2, max_clusters+1)
                    
                    for k in possible_k:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(df_scaled)
                        inertia.append(kmeans.inertia_)
                    
                    fig = px.line(x=list(possible_k), y=inertia, 
                                 title='Método del Codo', markers=True)
                    fig.update_layout(xaxis_title='Número de clusters', 
                                     yaxis_title='Inercia')
                    st.plotly_chart(fig)
                    
                    # Selección de K
                    n_clusters = st.slider("Número de clusters", 2, max_clusters, 3)
                    
                    # Aplicar K-Means
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(df_scaled)
                    
                    # Visualización
                    if len(selected_vars) >= 2:
                        df_vis = df_filled.copy()
                        df_vis['Cluster'] = clusters
                        
                        # Gráfico 2D
                        fig = px.scatter(df_vis, x=selected_vars[0], y=selected_vars[1],
                                       color='Cluster', title='Visualización de Clusters')
                        st.plotly_chart(fig)
                        
                        # Gráfico 3D si hay suficientes variables
                        if len(selected_vars) >= 3:
                            fig = px.scatter_3d(df_vis, x=selected_vars[0], y=selected_vars[1], z=selected_vars[2],
                                              color='Cluster', title='Visualización 3D de Clusters')
                            st.plotly_chart(fig)
                    
                    # Estadísticas por cluster
                    st.write("### Estadísticas por Cluster")
                    df_cluster_stats = df_filled.copy()
                    df_cluster_stats['Cluster'] = clusters
                    st.dataframe(df_cluster_stats.groupby('Cluster').mean().style.background_gradient(cmap='Blues'))
                    
                    # Silhouette Score
                    silhouette_avg = silhouette_score(df_scaled, clusters)
                    st.success(f"Silhouette Score: {silhouette_avg:.2f} (Valor entre -1 y 1, donde valores más altos indican mejor definición de clusters)")
        
        # ====================== Guardar Resultados ======================
        elif analysis_type == "Guardar Resultados":
            st.subheader("💾 Exportar Resultados")
            
            if 'model' in locals() or 'model' in globals():
                st.write("### Exportar Modelo Entrenado")
                model_name = st.text_input("Nombre del modelo", "mi_modelo")
                
                if st.button("Guardar Modelo"):
                    import joblib
                    joblib.dump(model, f'{model_name}.joblib')
                    st.success(f"Modelo guardado como {model_name}.joblib")
                    
                    # Descargar el modelo
                    with open(f'{model_name}.joblib', 'rb') as f:
                        st.download_button(
                            label="Descargar Modelo",
                            data=f,
                            file_name=f'{model_name}.joblib',
                            mime='application/octet-stream'
                        )
            
            if 'df' in locals() or 'df' in globals():
                st.write("### Exportar Datos Procesados")
                export_format = st.selectbox("Formato de exportación", ["CSV", "Excel"])
                export_name = st.text_input("Nombre del archivo", "datos_procesados")
                
                if st.button("Exportar Datos"):
                    if export_format == "CSV":
                        df.to_csv(f'{export_name}.csv', index=False)
                        with open(f'{export_name}.csv', 'rb') as f:
                            st.download_button(
                                label="Descargar CSV",
                                data=f,
                                file_name=f'{export_name}.csv',
                                mime='text/csv'
                            )
                    else:
                        df.to_excel(f'{export_name}.xlsx', index=False)
                        with open(f'{export_name}.xlsx', 'rb') as f:
                            st.download_button(
                                label="Descargar Excel",
                                data=f,
                                file_name=f'{export_name}.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            )
                    st.success("Datos exportados exitosamente")
    
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")

# Mensaje cuando no hay archivo cargado
else:
    st.info("👈 Por favor, sube un archivo CSV para comenzar el análisis")
    st.markdown("""
    ### Instrucciones:
    1. Sube un archivo CSV usando el panel izquierdo
    2. Selecciona el tipo de análisis que deseas realizar
    3. Explora los resultados y visualizaciones
    
    **Tipos de análisis disponibles:**
    - 📊 EDA: Análisis exploratorio de datos
    - 🔮 Modelos Predictivos: Regresión y clasificación
    - 🔎 PCA: Análisis de componentes principales
    - 🎭 MCA: Análisis de correspondencias múltiples
    - 🔍 Clustering: Segmentación de datos
    - 💾 Guardar Resultados: Exportar modelos y datos
    """)
