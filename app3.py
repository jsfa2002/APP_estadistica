# -*- coding: utf-8 -*-
"""ReDim - Análisis Multivariado Avanzado"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report, 
                           mean_squared_error, r2_score, silhouette_score)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from prince import MCA
from sklearn.pipeline import Pipeline

# Configuración de la página
st.set_page_config(page_title="ReDim - Análisis Multivariado", layout="wide", page_icon="📊")
st.title('📊 ReDim: Análisis Multivariado Avanzado')

# CSS personalizado mejorado
st.markdown("""
    <style>
    .stApp { background-color: #f9f9f9; }
    h1 { color: #2E86C1; text-align: center; font-weight: 700; }
    h2 { color: #3498DB; border-bottom: 2px solid #3498DB; padding-bottom: 5px; }
    h3 { color: #5DADE2; }
    .sidebar .sidebar-content { background-color: #EBF5FB; }
    .stButton>button { background-color: #2E86C1; color: white; }
    .stDownloadButton>button { background-color: #28B463; color: white; }
    .stAlert { border-left: 5px solid #2E86C1; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar para subir el archivo
st.sidebar.header("📂 Configuración de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo (CSV o Excel)", type=["csv", "xlsx"])

# Variables globales
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.sidebar.success("Archivo cargado exitosamente!")
    
    # Identificar tipos de columnas
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Mostrar vista previa de datos
    st.subheader("📌 Vista previa de los datos")
    with st.expander("Ver datos completos"):
        st.dataframe(df)
    
    # Mostrar información básica del dataset
    st.subheader("🔍 Metadatos del Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", df.shape[0])
        st.write("**Variables Numéricas:**", list(numeric_cols))
    with col2:
        st.metric("Total de Variables", df.shape[1])
        st.write("**Variables Categóricas:**", list(cat_cols))
    with col3:
        st.metric("Valores Faltantes", df.isnull().sum().sum())
        st.write("**Memoria Usada:**", f"{df.memory_usage().sum() / (1024*1024):.2f} MB")

    # Menú de análisis mejorado
    analysis_type = st.sidebar.selectbox(
        "Selecciona el tipo de análisis",
        ["EDA", "Modelos Predictivos", "Reducción de Dimensionalidad", "Análisis de Clusters"],
        index=0
    )

    # ====================== EDA MEJORADO ======================
    if analysis_type == "EDA":
        st.subheader("📊 Análisis Exploratorio de Datos (EDA) Avanzado")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Distribuciones", "Correlaciones", "Datos Faltantes"])
        
        with tab1:
            st.write("### Estadísticas Descriptivas")
            st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'))
            
            st.write("### Tipos de Datos")
            dtype_df = pd.DataFrame(df.dtypes.value_counts()).reset_index()
            dtype_df.columns = ['Tipo', 'Conteo']
            fig = px.pie(dtype_df, values='Conteo', names='Tipo', title='Distribución de Tipos de Datos')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if len(numeric_cols) > 0:
                selected_num = st.selectbox("Selecciona variable numérica", numeric_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df, x=selected_num, nbins=30, title=f'Distribución de {selected_num}')
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.box(df, y=selected_num, title=f'Boxplot de {selected_num}')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Análisis de outliers usando IQR
                Q1 = df[selected_num].quantile(0.25)
                Q3 = df[selected_num].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[selected_num] < lower_bound) | (df[selected_num] > upper_bound)]
                
                if not outliers.empty:
                    st.warning(f"⚠️ Se detectaron {len(outliers)} outliers en {selected_num}")
                    st.dataframe(outliers)
            
            if len(cat_cols) > 0:
                selected_cat = st.selectbox("Selecciona variable categórica", cat_cols)
                fig = px.bar(df[selected_cat].value_counts().reset_index(), 
                            x='count', y=selected_cat, 
                            title=f'Distribución de {selected_cat}',
                            orientation='h')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if len(numeric_cols) > 1:
                st.write("### Matriz de Correlación Numérica")
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, 
                              text_auto=True, 
                              aspect="auto",
                              color_continuous_scale='RdBu',
                              range_color=[-1, 1],
                              title='Matriz de Correlación')
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlaciones más altas y bajas
                corr_series = corr_matrix.unstack().sort_values(ascending=False)
                high_corr = corr_series[corr_series < 1].head(5)
                low_corr = corr_series.tail(5)
                
                st.write("**Correlaciones más altas:**")
                st.write(high_corr)
                
                st.write("**Correlaciones más bajas:**")
                st.write(low_corr)
        
        with tab4:
            st.write("### Análisis de Valores Faltantes")
            null_data = df.isnull().sum().reset_index()
            null_data.columns = ['Variable', 'Conteo Nulos']
            null_data['Porcentaje'] = (null_data['Conteo Nulos'] / len(df)) * 100
            
            fig = px.bar(null_data.sort_values('Porcentaje', ascending=False), 
                        x='Porcentaje', y='Variable', 
                        title='Porcentaje de Valores Faltantes por Variable',
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
            # Estrategias para manejar nulos
            if null_data['Conteo Nulos'].sum() > 0:
                st.warning("¡Advertencia: Hay valores faltantes en tus datos!")
                st.write("**Opciones para manejar nulos:**")
                st.markdown("""
                - **Eliminar filas:** `df.dropna()`
                - **Eliminar columnas:** `df.dropna(axis=1)`
                - **Imputar con media/mediana:** `df.fillna(df.mean())`
                - **Imputar con moda:** `df.fillna(df.mode().iloc[0])`
                - **Interpolación:** `df.interpolate()`
                """)

    # ====================== MODELOS PREDICTIVOS MEJORADOS ======================
    elif analysis_type == "Modelos Predictivos":
        st.subheader("🔮 Modelado Predictivo Avanzado")
        
        tab1, tab2, tab3 = st.tabs(["Configuración", "Entrenamiento", "Evaluación"])
        
        with tab1:
            target_var = st.selectbox("Selecciona la variable objetivo (Y)", df.columns)
            predictor_vars = st.multiselect("Selecciona las variables predictoras (X)", 
                                          df.columns.drop(target_var),
                                          default=list(df.columns.drop(target_var))[:3])
            
            # Determinar tipo de problema
            if len(df[target_var].unique()) <= 5 and df[target_var].dtype in ['object', 'category']:
                problem_type = "Clasificación"
                st.success("🔮 Se detectó un problema de Clasificación")
            else:
                problem_type = "Regresión"
                st.success("📈 Se detectó un problema de Regresión")
            
            # Selección de modelos según el tipo de problema
            if problem_type == "Clasificación":
                model_options = {
                    "Regresión Logística": LogisticRegression(max_iter=1000),
                    "Análisis Discriminante Lineal (LDA)": LinearDiscriminantAnalysis(),
                    "Análisis Discriminante Cuadrático (QDA)": QuadraticDiscriminantAnalysis(),
                    "Random Forest": RandomForestClassifier(),
                    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                }
            else:
                model_options = {
                    "Regresión Lineal": LinearRegression(),
                    "Regresión Ridge": Ridge(),
                    "Regresión Lasso": Lasso(),
                    "Random Forest": RandomForestRegressor(),
                    "XGBoost": XGBRegressor()
                }
            
            selected_models = st.multiselect("Selecciona modelos a comparar",
                                           list(model_options.keys()),
                                           default=list(model_options.keys())[:2])
            
            # Opciones avanzadas
            with st.expander("⚙️ Opciones Avanzadas"):
                test_size = st.slider("Tamaño del conjunto de prueba", 0.1, 0.5, 0.2, 0.05)
                random_state = st.number_input("Semilla aleatoria", 0, 1000, 42)
                cv_folds = st.number_input("Número de folds para validación cruzada", 2, 10, 5)
                
                # Hiperparámetros para GridSearch
                if st.checkbox("Optimizar hiperparámetros (GridSearch)"):
                    st.write("Configura los rangos para GridSearch")
                    param_grid = {}
                    for model_name in selected_models:
                        if model_name == "Random Forest":
                            param_grid[model_name] = {
                                'n_estimators': st.multiselect("n_estimators para Random Forest", 
                                                              [50, 100, 200], [100]),
                                'max_depth': st.multiselect("max_depth para Random Forest", 
                                                          [None, 5, 10], [None])
                            }
                        elif model_name == "XGBoost":
                            param_grid[model_name] = {
                                'learning_rate': st.multiselect("learning_rate para XGBoost", 
                                                               [0.01, 0.1, 0.3], [0.1]),
                                'max_depth': st.multiselect("max_depth para XGBoost", 
                                                          [3, 6, 9], [6])
                            }
        
        with tab2:
            if st.button("🚀 Entrenar Modelos"):
                st.session_state['model_results'] = {}
                
                X = df[predictor_vars]
                y = df[target_var]
                
                # Preprocesamiento
                if X.select_dtypes(include=['object']).any().any():
                    X = pd.get_dummies(X, drop_first=True)
                
                # Imputar valores faltantes
                imputer = SimpleImputer(strategy='mean' if problem_type == "Regresión" else 'most_frequent')
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                
                # Escalar características
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Dividir datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=random_state)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, model_name in enumerate(selected_models):
                    status_text.text(f"Entrenando {model_name}...")
                    progress_bar.progress((i + 1) / len(selected_models))
                    
                    model = model_options[model_name]
                    
                    # Entrenar modelo (con o sin GridSearch)
                    if 'param_grid' in locals() and model_name in param_grid:
                        grid_search = GridSearchCV(model, param_grid[model_name], 
                                                cv=cv_folds, scoring='accuracy' if problem_type == "Clasificación" else 'r2')
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                    else:
                        best_model = model
                        best_model.fit(X_train, y_train)
                        best_params = "No optimizado"
                    
                    # Evaluar modelo
                    y_pred = best_model.predict(X_test)
                    
                    if problem_type == "Clasificación":
                        accuracy = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred, output_dict=True)
                        cm = confusion_matrix(y_test, y_pred)
                        
                        st.session_state['model_results'][model_name] = {
                            'model': best_model,
                            'type': 'classification',
                            'accuracy': accuracy,
                            'report': report,
                            'confusion_matrix': cm,
                            'best_params': best_params
                        }
                    else:
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        st.session_state['model_results'][model_name] = {
                            'model': best_model,
                            'type': 'regression',
                            'mse': mse,
                            'r2': r2,
                            'best_params': best_params,
                            'y_test': y_test,
                            'y_pred': y_pred
                        }
                
                status_text.text("¡Entrenamiento completado!")
                progress_bar.empty()
                st.balloons()
        
        with tab3:
            if 'model_results' in st.session_state and st.session_state['model_results']:
                st.subheader("📊 Resultados de los Modelos")
                
                best_model_name = None
                best_score = -np.inf
                
                for model_name, results in st.session_state['model_results'].items():
                    with st.expander(f"🔍 {model_name}", expanded=True):
                        st.write(f"**Parámetros óptimos:** `{results['best_params']}`")
                        
                        if results['type'] == 'classification':
                            st.write(f"**Exactitud:** {results['accuracy']:.4f}")
                            
                            # Matriz de confusión
                            fig, ax = plt.subplots()
                            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel('Predicho')
                            ax.set_ylabel('Real')
                            ax.set_title(f'Matriz de Confusión - {model_name}')
                            st.pyplot(fig)
                            
                            # Reporte de clasificación
                            st.write("**Reporte de Clasificación:**")
                            report_df = pd.DataFrame(results['report']).transpose()
                            st.dataframe(report_df.style.background_gradient(cmap='Blues'))
                            
                            # Actualizar mejor modelo
                            if results['accuracy'] > best_score:
                                best_score = results['accuracy']
                                best_model_name = model_name
                        else:
                            st.write(f"**Error Cuadrático Medio (MSE):** {results['mse']:.4f}")
                            st.write(f"**Coeficiente de Determinación (R²):** {results['r2']:.4f}")
                            
                            # Gráfico de valores reales vs predichos
                            fig = px.scatter(x=results['y_test'], y=results['y_pred'], 
                                           labels={'x': 'Valores Reales', 'y': 'Valores Predichos'},
                                           title=f'Valores Reales vs Predichos - {model_name}')
                            fig.add_shape(type='line', x0=results['y_test'].min(), y0=results['y_test'].min(),
                                        x1=results['y_test'].max(), y1=results['y_test'].max(),
                                        line=dict(color='Red', dash='dash'))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Actualizar mejor modelo
                            if results['r2'] > best_score:
                                best_score = results['r2']
                                best_model_name = model_name
                
                if best_model_name:
                    st.success(f"🏆 Mejor modelo: {best_model_name} con {'exactitud' if problem_type == 'Clasificación' else 'R²'} de {best_score:.4f}")
                    
                    # Exportar modelo
                    import joblib
                    model_bytes = BytesIO()
                    joblib.dump(st.session_state['model_results'][best_model_name]['model'], model_bytes)
                    model_bytes.seek(0)
                    
                    st.download_button(
                        label="⬇️ Descargar mejor modelo",
                        data=model_bytes,
                        file_name=f"best_model_{best_model_name.replace(' ', '_')}.pkl",
                        mime="application/octet-stream"
                    )

    # ====================== REDUCCIÓN DE DIMENSIONALIDAD ======================
    elif analysis_type == "Reducción de Dimensionalidad":
        st.subheader("🔎 Reducción de Dimensionalidad")
        
        dim_method = st.radio("Selecciona el método", 
                            ["PCA (Análisis de Componentes Principales)", 
                             "MCA (Análisis de Correspondencias Múltiples)"])
        
        if dim_method == "PCA (Análisis de Componentes Principales)":
            if len(numeric_cols) < 2:
                st.error("Se necesitan al menos 2 variables numéricas para ejecutar PCA")
            else:
                # Selección de variables
                selected_vars = st.multiselect("Selecciona variables para PCA", 
                                             numeric_cols, default=list(numeric_cols))
                
                # Opciones de PCA
                n_components = st.slider("Número de componentes", 2, min(10, len(selected_vars)), 2)
                scale_data = st.checkbox("Estandarizar datos", value=True)
                
                if st.button("Ejecutar PCA"):
                    # Preprocesamiento
                    df_pca = df[selected_vars].copy()
                    imputer = SimpleImputer(strategy='mean')
                    df_filled = pd.DataFrame(imputer.fit_transform(df_pca), columns=selected_vars)
                    
                    # Escalar si es necesario
                    if scale_data:
                        scaler = StandardScaler()
                        df_scaled = scaler.fit_transform(df_filled)
                    else:
                        df_scaled = df_filled.values
                    
                    # Aplicar PCA
                    pca = PCA(n_components=n_components)
                    pca.fit(df_scaled)
                    components = pca.transform(df_scaled)
                    
                    # Resultados
                    st.write("### Varianza explicada")
                    var_exp = pca.explained_variance_ratio_
                    cum_var_exp = np.cumsum(var_exp)
                    
                    fig = px.bar(x=range(1, n_components+1), y=var_exp,
                                labels={'x': 'Componente', 'y': 'Varianza explicada'},
                                title='Varianza explicada por componente')
                    fig.add_scatter(x=range(1, n_components+1), y=cum_var_exp, 
                                   mode='lines+markers', name='Acumulada')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Gráfico de componentes
                    st.write("### Gráfico de Componentes Principales")
                    pc_df = pd.DataFrame(components[:, :2], columns=['PC1', 'PC2'])
                    
                    # Añadir variables categóricas para colorear si existen
                    if len(cat_cols) > 0:
                        color_var = st.selectbox("Variable para colorear puntos", cat_cols)
                        pc_df[color_var] = df[color_var].values
                        
                        fig = px.scatter(pc_df, x='PC1', y='PC2', color=color_var,
                                        title='PCA: Componente 1 vs Componente 2',
                                        hover_data={color_var: True})
                    else:
                        fig = px.scatter(pc_df, x='PC1', y='PC2', 
                                       title='PCA: Componente 1 vs Componente 2')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cargas factoriales
                    st.write("### Cargas Factoriales")
                    loadings = pd.DataFrame(
                        pca.components_.T,
                        columns=[f'PC{i+1}' for i in range(n_components)],
                        index=selected_vars
                    )
                    st.dataframe(loadings.style.background_gradient(cmap='RdBu', axis=None, vmin=-1, vmax=1))
                    
                    # Exportar resultados
                    csv = loadings.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        "⬇️ Descargar cargas factoriales",
                        csv,
                        "pca_loadings.csv",
                        "text/csv"
                    )
        
        else:  # MCA
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
                    st.write("### Inercia explicada")
                    fig = px.bar(x=range(1, n_components+1), y=explained_inertia,
                                labels={'x': 'Dimensión', 'y': 'Proporción de inercia explicada'},
                                title='Inercia explicada por dimensión')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Gráfico de categorías
                    st.write("### Gráfico de Categorías")
                    col_coords_df = mca_col_coords.reset_index()
                    col_coords_df['Variable'] = col_coords_df['index'].str.split('_').str[0]
                    
                    fig = px.scatter(col_coords_df, x=0, y=1, color='Variable',
                                   hover_name='index', 
                                   labels={'0': 'Dimensión 1', '1': 'Dimensión 2'},
                                   title='MCA: Representación de categorías')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Exportar coordenadas
                    csv = mca_col_coords.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        "⬇️ Descargar coordenadas MCA",
                        csv,
                        "mca_coordinates.csv",
                        "text/csv"
                    )

    # ====================== ANÁLISIS DE CLUSTERS ======================
    elif analysis_type == "Análisis de Clusters":
        st.subheader("👥 Análisis de Clusters")
        
        cluster_method = st.selectbox("Selecciona el método de clustering",
                                    ["K-Means", "DBSCAN"])
        
        # Selección de variables
        selected_vars = st.multiselect("Selecciona variables para clustering", 
                                     numeric_cols, default=list(numeric_cols)[:3])
        
        # Preprocesamiento
        df_cluster = df[selected_vars].copy()
        imputer = SimpleImputer(strategy='mean')
        df_filled = pd.DataFrame(imputer.fit_transform(df_cluster), columns=selected_vars)
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_filled)
        
        if cluster_method == "K-Means":
            # Método del codo para determinar k
            st.write("### Método del Codo para Determinar K Óptimo")
            distortions = []
            max_clusters = min(10, len(df_scaled)-1)
            K = range(1, max_clusters+1)
            
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(df_scaled)
                distortions.append(kmeans.inertia_)
            
            fig = px.line(x=K, y=distortions, 
                         labels={'x': 'Número de clusters', 'y': 'Inercia'},
                         title='Método del Codo')
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig, use_container_width=True)
            
            # Seleccionar número de clusters
            n_clusters = st.slider("Número de clusters", 2, max_clusters, 3)
            
            if st.button("Ejecutar K-Means"):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(df_scaled)
                
                # Añadir clusters al dataframe
                df_cluster['Cluster'] = clusters
                
                # Visualización (usando PCA si hay más de 2 variables)
                if len(selected_vars) > 2:
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(df_scaled)
                    plot_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
                    plot_df['Cluster'] = clusters
                    
                    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                                   title='Visualización de Clusters (PCA)')
                else:
                    plot_df = df_cluster.copy()
                    fig = px.scatter(plot_df, x=selected_vars[0], y=selected_vars[1], 
                                   color='Cluster', title='Visualización de Clusters')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Estadísticas por cluster
                st.write("### Estadísticas por Cluster")
                cluster_stats = df_cluster.groupby('Cluster').mean()
                st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))
                
                # Silhouette score
                silhouette_avg = silhouette_score(df_scaled, clusters)
                st.write(f"**Silhouette Score:** {silhouette_avg:.3f}")
                st.caption("Valores cercanos a 1 indican clusters bien definidos.")
                
                # Exportar resultados
                csv = df_cluster.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Descargar datos con clusters",
                    csv,
                    "data_with_clusters.csv",
                    "text/csv"
                )
        
        else:  # DBSCAN
            st.write("### Configuración de DBSCAN")
            eps = st.slider("EPS (Distancia máxima entre puntos)", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Mínimo de muestras por cluster", 1, 20, 5)
            
            if st.button("Ejecutar DBSCAN"):
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(df_scaled)
                
                # Añadir clusters al dataframe
                df_cluster['Cluster'] = clusters
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                n_noise = list(clusters).count(-1)
                
                st.write(f"**Número de clusters encontrados:** {n_clusters}")
                st.write(f"**Puntos considerados ruido:** {n_noise}")
                
                # Visualización
                if len(selected_vars) > 2:
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(df_scaled)
                    plot_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
                    plot_df['Cluster'] = clusters
                    
                    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster',
                                   title='Visualización de Clusters DBSCAN (PCA)')
                else:
                    plot_df = df_cluster.copy()
                    fig = px.scatter(plot_df, x=selected_vars[0], y=selected_vars[1], 
                                   color='Cluster', title='Visualización de Clusters DBSCAN')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Exportar resultados
                csv = df_cluster.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Descargar datos con clusters",
                    csv,
                    "data_with_clusters_dbscan.csv",
                    "text/csv"
                )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
🚀 **ReDim Team**   
""")

# Notas finales
with st.expander("ℹ️ Acerca de esta aplicación"):
    st.markdown("""
    **ReDim - Análisis Multivariado Avanzado**  
    Versión 2.0 | Mayo 2025  
    
    Esta aplicación permite realizar:
    - Análisis exploratorio de datos (EDA)
    - Modelado predictivo (clasificación y regresión)
    - Reducción de dimensionalidad (PCA, MCA)
    - Análisis de clusters (K-Means, DBSCAN)
    
    Desarrollado con Python, Streamlit y Scikit-learn.
    """)

