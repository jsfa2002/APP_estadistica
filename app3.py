# Añade al inicio del script (con los otros imports)
from collections import defaultdict

# Después de cargar el dataframe (en el if uploaded_file is not None)
if uploaded_file is not None:
    # Diccionario para almacenar resultados de modelos
    if 'model_results' not in st.session_state:
        st.session_state.model_results = defaultdict(dict)
    
    # [...] (todo el código anterior se mantiene igual hasta la sección de Modelos Predictivos)

    # ====================== MODELOS PREDICTIVOS ======================
    elif analysis_type == "Modelos Predictivos":
        st.subheader("🔮 Modelos Predictivos")
        
        # Pestañas para organización
        tab1, tab2, tab3 = st.tabs(["🏋️ Entrenar Modelo", "🔍 Comparar Modelos", "📊 Comparar Todos"])
        
        with tab1:
            # [...] (todo el código de entrenamiento anterior se mantiene igual)
            
            # MODIFICACIÓN: Al final del entrenamiento, guardamos los resultados
            if st.button("Ejecutar Modelo"):
                # [...] (código de entrenamiento anterior)
                
                # Guardamos resultados en session_state
                model_key = f"{model_choice}_{target_var}"
                st.session_state.model_results[model_key] = {
                    'model_type': model_type,
                    'metrics': {
                        'MSE': mse if model_type == "regression" else None,
                        'R2': r2 if model_type == "regression" else None,
                        'Accuracy': accuracy if model_type == "classification" else None,
                        'Precision': precision if model_type == "classification" else None,
                        'Recall': recall if model_type == "classification" else None,
                        'F1': f1 if model_type == "classification" else None
                    },
                    'model': model,
                    'predictors': predictor_vars,
                    'target': target_var,
                    'timestamp': pd.Timestamp.now()
                }
                
                st.success(f"Modelo {model_choice} guardado para comparación!")
        
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
                    default=available_models[:2]
                )
                
                if selected_models:
                    # Filtramos solo los modelos seleccionados
                    results_to_compare = {k: v for k, v in st.session_state.model_results.items() 
                                        if k in selected_models}
                    
                    # Creamos dataframe comparativo
                    comparison_data = []
                    for model_name, model_data in results_to_compare.items():
                        row = {'Modelo': model_name}
                        row.update(model_data['metrics'])
                        comparison_data.append(row)
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    # Mostramos tabla comparativa
                    st.write("### Métricas Comparativas")
                    st.dataframe(df_comparison.style.format("{:.4f}").highlight_max(axis=0))
                    
                    # Gráfico comparativo
                    st.write("### Visualización Comparativa")
                    
                    # Seleccionamos qué métricas mostrar
                    if len(selected_models) > 0:
                        model_types = set(st.session_state.model_results[m]['model_type'] 
                                    for m in selected_models)
                        
                        if len(model_types) == 1:  # Todos son del mismo tipo
                            if "regression" in model_types:
                                metrics_to_show = ['MSE', 'R2']
                            else:
                                metrics_to_show = ['Accuracy', 'Precision', 'Recall', 'F1']
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            df_comparison.set_index('Modelo')[metrics_to_show].plot(
                                kind='bar', ax=ax)
                            ax.set_title("Comparación de Modelos")
                            ax.set_ylabel("Valor Métrica")
                            st.pyplot(fig)
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
                    
                    df_reg = pd.DataFrame(reg_data)
                    st.dataframe(df_reg.style.format("{:.4f}").highlight_max(axis=0))
                    
                    # Gráfico de regresión
                    if len(regression_models) > 1:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        df_reg.set_index('Modelo').plot(kind='bar', ax=ax)
                        ax.set_title("Comparación de Modelos de Regresión")
                        st.pyplot(fig)
                
                # Comparación de modelos de clasificación
                if classification_models:
                    st.write("### 🏷️ Modelos de Clasificación")
                    clf_data = []
                    for model_name, model_data in classification_models.items():
                        row = {'Modelo': model_name}
                        row.update({k: v for k, v in model_data['metrics'].items() if v is not None})
                        clf_data.append(row)
                    
                    df_clf = pd.DataFrame(clf_data)
                    st.dataframe(df_clf.style.format("{:.4f}").highlight_max(axis=0))
                    
                    # Gráfico de clasificación
                    if len(classification_models) > 1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df_clf.set_index('Modelo').plot(kind='bar', ax=ax)
                        ax.set_title("Comparación de Modelos de Clasificación")
                        st.pyplot(fig)
                
                if not regression_models and not classification_models:
                    st.warning("No hay modelos para comparar")
