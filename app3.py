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
st.set_page_config(page_title="ReDim - Análisis Multivariado y Métodos Predictivos", layout="wide")
st.title('📊 ReDim: Análisis Multivariado y Métodos Predictivos')

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
        # ✅ Agregamos tab4 desde el inicio
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏋️ Entrenar Modelo",
            "🔍 Comparar Modelos",
            "📊 Comparar Todos",
            "🔮 Predicción Manual"])
      
        
        with tab1:
            model_choice = st.selectbox("Selecciona el modelo", 
                                        ["Regresión Lineal", "Regresión Logística", "LDA", "QDA", 
                                         "Árbol de Decisión (Clasificación)", "Árbol de Decisión (Regresión)",
                                         "Random Forest (Clasificación)", "Random Forest (Regresión)"])
        
            target_var = st.selectbox("Selecciona la variable dependiente (Y)", df.columns)
            predictor_vars = st.multiselect("Selecciona las variables predictoras (X)", df.columns.drop(target_var))
        
            with st.expander("⚙ Configuración Avanzada"):
                test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 40, 30)
                random_state = st.number_input("Semilla aleatoria", value=42)
        
                if "Árbol" in model_choice or "Forest" in model_choice:
                    max_depth = st.number_input("Profundidad máxima del árbol", min_value=1, max_value=20, value=3)
                    min_samples_split = st.number_input("Mínimo de muestras para dividir", min_value=2, max_value=20, value=2)
                    if "Forest" in model_choice:
                        n_estimators = st.number_input("Número de árboles", min_value=10, max_value=500, value=100)
        
            if st.button("🔧 Entrenar Modelo"):
                st.subheader(f"📈 Resultados del Modelo: {model_choice}")
        
                X = df[predictor_vars]
                y = df[target_var]
        
                if X.select_dtypes(include=['object']).any().any():
                    X = pd.get_dummies(X, drop_first=True)
        
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)
        
                model = None
                model_type = ""
        
                if model_choice == "Regresión Lineal":
                    model = LinearRegression()
                    model_type = "regression"
                elif model_choice == "Regresión Logística":
                    model = LogisticRegression(max_iter=1000, random_state=random_state)
                    model_type = "classification"
                elif model_choice == "LDA":
                    model = LinearDiscriminantAnalysis()
                    model_type = "classification"
                elif model_choice == "QDA":
                    model = QuadraticDiscriminantAnalysis()
                    model_type = "classification"
                elif model_choice == "Árbol de Decisión (Clasificación)":
                    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
                    model_type = "classification"
                elif model_choice == "Árbol de Decisión (Regresión)":
                    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
                    model_type = "regression"
                elif model_choice == "Random Forest (Clasificación)":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
                    model_type = "classification"
                elif model_choice == "Random Forest (Regresión)":
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
                    model_type = "regression"
        
                with st.spinner("Entrenando modelo..."):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
        
                if model_type == "regression":
                    st.write("### Métricas de Regresión")
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
        
                    metrics_df = pd.DataFrame({
                        "Métrica": ["Error Cuadrático Medio (MSE)", "Raíz del Error Cuadrático Medio (RMSE)", 
                                    "Coeficiente de Determinación (R²)"],
                        "Valor": [mse, rmse, r2]
                    })
                    st.dataframe(metrics_df.style.format({"Valor": "{:.4f}"}))
        
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred, alpha=0.5)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                    ax.set_xlabel('Valores Reales')
                    ax.set_ylabel('Valores Predichos')
                    ax.set_title('Valores Reales vs Predichos')
                    st.pyplot(fig)
        
                    if hasattr(model, 'coef_'):
                        st.write("### Coeficientes Beta (β)")
                        coef_df = pd.DataFrame({
                            "Variable": X.columns,
                            "Coeficiente (β)": model.coef_.flatten()
                        })
        
                        st.markdown("""<div style="overflow-x: auto;"><table class="coef-table">
                                        <thead><tr><th>Variable</th><th>Coeficiente (β)</th></tr></thead><tbody>""", unsafe_allow_html=True)
        
                        for _, row in coef_df.iterrows():
                            st.markdown(f"<tr><td>{row['Variable']}</td><td>{row['Coeficiente (β)']:.6f}</td></tr>", unsafe_allow_html=True)
        
                        st.markdown("""</tbody></table></div>""", unsafe_allow_html=True)
        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Coeficiente (β)', y='Variable', data=coef_df.sort_values('Coeficiente (β)', ascending=False), ax=ax)
                        ax.set_title('Magnitud de los Coeficientes Beta')
                        st.pyplot(fig)
        
                else:
                    st.write("### Métricas de Clasificación")
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
        
                    metrics_df = pd.DataFrame({
                        "Métrica": ["Exactitud (Accuracy)", "Precisión (Precision)", 
                                    "Sensibilidad (Recall)", "F1-Score"],
                        "Valor": [accuracy, precision, recall, f1]
                    })
                    st.dataframe(metrics_df.style.format({"Valor": "{:.4f}"}))
        
                    st.write("### Matriz de Confusión")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicho')
                    ax.set_ylabel('Real')
                    st.pyplot(fig)
        
                    st.write("### Reporte de Clasificación")
                    st.text(classification_report(y_test, y_pred))
        
                    # 🔵 Curva ROC y AUC
                    if hasattr(model, "predict_proba"):
                        with st.expander("📈 Curva ROC y AUC (opcional)"):
                            try:
                                y_pred_proba = model.predict_proba(X_test)[:, 1]
                                auc_score = roc_auc_score(y_test, y_pred_proba)
                    
                                st.write("### Área bajo la Curva ROC (AUC)")
                                st.metric(label="AUC", value=f"{auc_score:.4f}")
                    
                                fig, ax = plt.subplots(figsize=(8, 6))
                                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                                ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
                                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                                ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
                                ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
                                ax.set_title('Curva ROC')
                                ax.legend(loc="lower right")
                                st.pyplot(fig)
                    
                            except Exception as e:
                                st.warning(f"No se pudo calcular la Curva ROC: {e}")
        
                    if hasattr(model, 'coef_'):
                        st.write("### Coeficientes Beta (β)")
                        coef_df = pd.DataFrame({
                            "Variable": X.columns,
                            "Coeficiente (β)": model.coef_[0]
                        })
        
                        st.markdown("""<div style="overflow-x: auto;"><table class="coef-table">
                                        <thead><tr><th>Variable</th><th>Coeficiente (β)</th></tr></thead><tbody>""", unsafe_allow_html=True)
        
                        for _, row in coef_df.iterrows():
                            st.markdown(f"<tr><td>{row['Variable']}</td><td>{row['Coeficiente (β)']:.6f}</td></tr>", unsafe_allow_html=True)
        
                        st.markdown("""</tbody></table></div>""", unsafe_allow_html=True)
        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Coeficiente (β)', y='Variable', data=coef_df.sort_values('Coeficiente (β)', ascending=False), ax=ax)
                        ax.set_title('Magnitud de los Coeficientes Beta')
                        st.pyplot(fig)
        
                if "Árbol" in model_choice and not "Forest" in model_choice:
                    st.write("### Visualización del Árbol de Decisión")
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(model, 
                              feature_names=X.columns, 
                              class_names=[str(c) for c in model.classes_] if model_type == "classification" else None,
                              filled=True, rounded=True, ax=ax)
                    st.pyplot(fig)
        
                model_key = f"{model_choice} - {target_var} ({datetime.now().strftime('%H:%M:%S')})"
                st.session_state.model_results[model_key] = {
                    'model_type': model_type,
                    'metrics': {
                        'MSE': mse if model_type == "regression" else None,
                        'RMSE': rmse if model_type == "regression" else None,
                        'R2': r2 if model_type == "regression" else None,
                        'Accuracy': accuracy if model_type == "classification" else None,
                        'Precision': precision if model_type == "classification" else None,
                        'Recall': recall if model_type == "classification" else None,
                        'F1': f1 if model_type == "classification" else None
                    },
                    'model': model,
                    'predictors': predictor_vars,
                    'target': target_var,
                    'X_test': X_test,
                    'y_test': y_test,
                    'params': {
                        'test_size': test_size,
                        'random_state': random_state,
                        'max_depth': max_depth if "Árbol" in model_choice or "Forest" in model_choice else None,
                        'min_samples_split': min_samples_split if "Árbol" in model_choice or "Forest" in model_choice else None,
                        'n_estimators': n_estimators if "Forest" in model_choice else None,
                        'coefficients': model.coef_.tolist() if hasattr(model, 'coef_') else None
                    },
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        
                st.session_state.last_model = model_key
                st.success(f"Modelo {model_choice} entrenado y guardado para comparación!")
        
                st.markdown(f"""
                    <div class="model-card">
                        <h4>Resumen del Modelo</h4>
                        <p><strong>Nombre:</strong> {model_key}</p>
                        <p><strong>Tipo:</strong> {'Regresión' if model_type == 'regression' else 'Clasificación'}</p>
                        <p><strong>Variables predictoras:</strong> {', '.join(predictor_vars)}</p>
                        <p><strong>Variable objetivo:</strong> {target_var}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("🔍 Comparar Modelos Individuales")
            
            if not st.session_state.model_results:
                st.warning("No hay modelos entrenados para comparar")
            else:
                # Selección de modelos a comparar
                available_models = list(st.session_state.model_results.keys())
                selected_models = st.multiselect(
                    "Selecciona modelos para comparar",
                    available_models,
                    default=[st.session_state.last_model] if st.session_state.last_model else available_models[:2]
                )
                
                if selected_models:
                    # Filtrar los modelos seleccionados
                    results_to_compare = {k: v for k, v in st.session_state.model_results.items() if k in selected_models}
                    
                    # Mostrar detalles de los modelos seleccionados
                    st.write("### Modelos Seleccionados")
                    cols = st.columns(len(selected_models))
                    for idx, model_name in enumerate(selected_models):
                        model_data = st.session_state.model_results[model_name]
                        with cols[idx]:
                            st.markdown(f"""
                            <div class="model-card">
                                <h5>{model_name}</h5>
                                <p><strong>Tipo:</strong> {'Regresión' if model_data['model_type'] == 'regression' else 'Clasificación'}</p>
                                <p><strong>Objetivo:</strong> {model_data['target']}</p>
                                <p><strong>Predictores:</strong> {len(model_data['predictors'])}</p>
                                <p><strong>Entrenado:</strong> {model_data['timestamp']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Crear dataframe comparativo
                    comparison_data = []
                    auc_scores = []
                    for model_name, model_data in results_to_compare.items():
                        row = {'Modelo': model_name}
                        for k, v in model_data['metrics'].items():
                            if v is not None:
                                if 'p-value' in k:
                                    v = round(v * 100, 2)
                                    row[k] = f"{v}%"
                                else:
                                    row[k] = v
                                    
                        if model_data['model_type'] == 'classification':
                            y_true = model_data['y_test']
                            y_pred_proba = model_data['model'].predict_proba(model_data['X_test'])[:, 1]
                            
                            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                            roc_auc = auc(fpr, tpr)
                            auc_scores.append(roc_auc)
                            row['AUC'] = round(roc_auc, 4)
                            
                        comparison_data.append(row)
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    # Mostrar tabla comparativa
                    st.write("### Métricas Comparativas")
                    st.dataframe(df_comparison.set_index('Modelo').style.format("{:.4f}").highlight_max(axis=0))
                    
                    # Gráfico comparativo
                    st.write("### Visualización Comparativa")
                    model_types = set(st.session_state.model_results[m]['model_type'] for m in selected_models)
                    
                    if len(model_types) == 1:
                        if "regression" in model_types:
                            metrics_to_show = ['MSE', 'RMSE', 'R2']
                            title = "Comparación de Modelos de Regresión"
                        else:
                            metrics_to_show = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
                            title = "Comparación de Modelos de Clasificación"
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df_comparison.set_index('Modelo')[metrics_to_show].plot(kind='bar', ax=ax)
                        ax.set_title(title)
                        ax.set_ylabel("Valor Métrica")
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        st.pyplot(fig)
                        
                        # Mostrar p-valores para modelos de regresión lineal
                        linear_models = [m for m in selected_models if hasattr(st.session_state.model_results[m]['model'], 'pvalues')]
                        
                        if linear_models:
                            st.write("### Comparación de P-Valores")
                            
                            all_pvalues = []
                            common_vars = None
                            
                            for model_name in linear_models:
                                model_data = st.session_state.model_results[model_name]
                                pvalues = model_data['model'].pvalues
                                
                                if common_vars is None:
                                    common_vars = set(pvalues.index)
                                else:
                                    common_vars &= set(pvalues.index)
                                
                                all_pvalues.append({
                                    'Modelo': model_name,
                                    'pvalues': pvalues
                                })
                            
                            if common_vars:
                                pvalue_comparison = []
                                for var in common_vars:
                                    row = {'Variable': var}
                                    for model_info in all_pvalues:
                                        model_name = model_info['Modelo']
                                        pval = model_info['pvalues'].get(var, None)
                                        if pval is not None:
                                            row[model_name] = f"{round(pval * 100, 2)}%"
                                        else:
                                            row[model_name] = "N/A"
                                    pvalue_comparison.append(row)
                                
                                df_pvalue_comparison = pd.DataFrame(pvalue_comparison).set_index('Variable')
                                
                                st.write("P-Valores para variables comunes:")
                                st.dataframe(df_pvalue_comparison)
                                
                                # Gráfico de p-valores
                                st.write("### Comparación Visual de P-Valores")
                                fig, ax = plt.subplots(figsize=(10, 6))
                                df_pvalue_comparison.applymap(lambda x: float(x.strip('%')) if isinstance(x, str) and '%' in x else None).plot(kind='bar', ax=ax)
                                ax.set_title("Comparación de P-Valores")
                                ax.set_ylabel("P-Valor (%)")
                                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                                st.pyplot(fig)
                            else:
                                st.warning("No hay variables comunes para comparar p-valores")
                        else:
                            st.warning("Ninguno de los modelos tiene p-valores disponibles para comparar")
                    else:
                        st.warning("Los modelos seleccionados son de tipos diferentes (regresión/clasificación)")
        
        with tab3:
            st.subheader("📊 Comparación Automática de Todos los Modelos")
            
            if not st.session_state.model_results:
                st.warning("No hay modelos entrenados para comparar")
            else:
                # Separamos modelos por tipo
                regression_models = {k: v for k, v in st.session_state.model_results.items() 
                                   if v['model_type'] == 'regression'}
                classification_models = {k: v for k, v in st.session_state.model_results.items() 
                                       if v['model_type'] == 'classification'}
                
                # Comparación de modelos de regresión
                if regression_models:
                    st.write("### 🔢 Modelos de Regresión")
                    reg_data = []
                    for model_name, model_data in regression_models.items():
                        row = {'Modelo': model_name}
                        row.update({k: v for k, v in model_data['metrics'].items() if v is not None})
                        reg_data.append(row)
                    
                    df_reg = pd.DataFrame(reg_data).set_index('Modelo')
                    st.dataframe(df_reg.style.format("{:.4f}").highlight_max(axis=0))
                    
                    # Gráfico de regresión
                    if len(regression_models) > 1:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        df_reg.plot(kind='bar', ax=ax)
                        ax.set_title("Comparación de Modelos de Regresión")
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        st.pyplot(fig)
                
                # Comparación de modelos de clasificación
                if classification_models:
                    st.write("### 🏷️ Modelos de Clasificación")
                    clf_data = []
                    for model_name, model_data in classification_models.items():
                        row = {'Modelo': model_name}
                        row.update({k: v for k, v in model_data['metrics'].items() if v is not None})
                        
                        # Calcular AUC si es posible
                        if hasattr(model_data['model'], 'predict_proba'):
                            try:
                                y_pred_proba = model_data['model'].predict_proba(model_data['X_test'])[:, 1]
                                fpr, tpr, _ = roc_curve(model_data['y_test'], y_pred_proba)
                                roc_auc = auc(fpr, tpr)
                                row['AUC'] = roc_auc
                            except:
                                row['AUC'] = None
                        
                        clf_data.append(row)
                    
                    df_clf = pd.DataFrame(clf_data).set_index('Modelo')
                    st.dataframe(df_clf.style.format("{:.4f}").highlight_max(axis=0))
                    
                    # Gráfico de clasificación
                    if len(classification_models) > 1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
                        if 'AUC' in df_clf.columns:
                            metrics.append('AUC')
                        df_clf[metrics].plot(kind='bar', ax=ax)
                        ax.set_title("Comparación de Modelos de Clasificación")
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        st.pyplot(fig)
                
                if not regression_models and not classification_models:
                    st.warning("No hay modelos para comparar")
            with tab4:
                st.subheader("🔮 Predicción Manual")
        
                if not st.session_state.model_results:
                    st.warning("No hay modelos entrenados para realizar predicciones.")
                else:
                    available_models = {
                        k: v for k, v in st.session_state.model_results.items()
                        if v['model_type'] in ['regression', 'classification']
                    }
        
                    if not available_models:
                        st.warning("Por ahora solo puedes predecir con modelos de regresión lineal o logística.")
                    else:
                        selected_model = st.selectbox("Selecciona un modelo para predecir", list(available_models.keys()))
                        model_data = st.session_state.model_results[selected_model]
        
                        st.markdown(f"Modelo seleccionado: *{selected_model}*")
                        predictors = model_data['predictors']
                        coefficients = model_data['params']['coefficients']
        
                        st.write("### Ingresa los valores de las variables predictoras")
                        user_inputs = {}
                        for predictor in predictors:
                            user_inputs[predictor] = st.number_input(f"{predictor}", value=0.0)
        
                        if st.button("🚀 Predecir"):
                            beta_0 = model_data['model'].intercept_
                            beta_rest = coefficients
                        
                            # Si beta_rest es lista de listas (por ejemplo en clasificación multiclase), tomamos la primera
                            if isinstance(beta_rest[0], list):
                                beta_rest = beta_rest[0]
                        
                            # Calcular z = β0 + β1·x1 + β2·x2 + ...
                            z = beta_0
                            for b, var in zip(beta_rest, predictors):
                                z += b * user_inputs[var]
                            
                            z = float(z)  # Asegurarse que sea escalar
                        
                            st.write(f"*Valor de Z:* {z:.4f}")
                        
                            # Predicción final
                            if model_data['model_type'] == 'regression':
                                st.success(f"🔵 Predicción de Regresión Lineal: {z:.4f}")
                            elif model_data['model_type'] == 'classification':
                                prob = 1 / (1 + np.exp(-z))
                                prob = float(prob)  # Convertir a escalar para evitar errores de formato
                                st.success(f"🟢 Probabilidad predicha (clase 1): {prob:.4f}")
                                st.info(f"🔵 Probabilidad clase 0: {(1 - prob):.4f}")
                                    
    
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
                
                fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                ax[0].bar(range(1, n_components+1), var_exp, alpha=0.6, align='center')
                ax[0].set_ylabel('Varianza explicada')
                ax[0].set_xlabel('Componente principal')
                ax[0].set_title('Varianza explicada por componente')
                
                ax[1].plot(range(1, n_components+1), cum_var_exp, 'o-')
                ax[1].set_ylabel('Varianza explicada acumulada')
                ax[1].set_xlabel('Componente principal')
                ax[1].set_title('Varianza acumulada')
                st.pyplot(fig)
                
                # Gráfico de componentes
                st.write("### Gráfico de los dos primeros componentes")
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(components[:, 0], components[:, 1], alpha=0.6)
                ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
                ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')
                ax.set_title('PCA: Componente 1 vs Componente 2')
                
                # Si hay una variable categórica, usarla para colorear
                if len(cat_cols) > 0:
                    color_var = st.selectbox("Variable para colorear puntos", cat_cols)
                    unique_cats = df[color_var].unique()
                    colors = plt.cm.get_cmap('tab10', len(unique_cats))
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    for i, cat in enumerate(unique_cats):
                        idx = df[color_var] == cat
                        ax.scatter(components[idx, 0], components[idx, 1], 
                                   color=colors(i), label=cat, alpha=0.6)
                    ax.legend(title=color_var)
                    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
                    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')
                    ax.set_title('PCA coloreado por ' + color_var)
                
                st.pyplot(fig)
                
                # Cargas factoriales
                st.write("### Cargas factoriales (Componentes principales)")
                loadings = pd.DataFrame(
                    pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(n_components)],
                    index=selected_vars
                )
                st.write(loadings)
                
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
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(range(1, n_components+1), explained_inertia, alpha=0.6)
                ax.set_xlabel('Dimensión')
                ax.set_ylabel('Proporción de inercia explicada')
                st.pyplot(fig)
                
                # Gráfico de individuos
                st.write("### Gráfico de Individuos (dos primeras dimensiones)")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(mca_coords.iloc[:, 0], mca_coords.iloc[:, 1], alpha=0.5, color='blue')
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel(f"Dimensión 1 ({explained_inertia[0]*100:.1f}%)")
                ax.set_ylabel(f"Dimensión 2 ({explained_inertia[1]*100:.1f}%)")
                ax.set_title("MCA: Individuos")
                st.pyplot(fig)
                
                # Gráfico de categorías
                st.write("### Gráfico de Categorías (dos primeras dimensiones)")
                fig, ax = plt.subplots(figsize=(10, 8))
                
                colors = plt.cm.get_cmap('tab10', len(selected_cats))
                for i, var in enumerate(selected_cats):
                    idx = mca_col_coords.index.str.startswith(var)
                    ax.scatter(mca_col_coords.iloc[idx, 0], mca_col_coords.iloc[idx, 1], 
                              color=colors(i), label=var, alpha=0.7)
                
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel(f"Dimensión 1 ({explained_inertia[0]*100:.1f}%)")
                ax.set_ylabel(f"Dimensión 2 ({explained_inertia[1]*100:.1f}%)")
                ax.set_title("MCA: Categorías")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)
                
                st.success("MCA ejecutado correctamente")

st.sidebar.markdown("---")
st.sidebar.markdown("🚀 Desarrollado con cariño por ReDim Team")
