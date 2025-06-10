#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de entrenamiento para detección de fraude en transacciones financieras.

Este script entrena múltiples modelos de Machine Learning utilizando diferentes conjuntos
de datos (original, SMOTE, ADASYN, SMOTETomek) y evalúa su rendimiento. Guarda los
resultados y los modelos entrenados.

Autor: Hlieb Sydorenko
Fecha: 10/06/2025
TFG: Comparativa de Modelos de Aprendizaje Automático y Estrategias de Aumentación de Datos para la Detección de Fraude en Transacciones Financieras
"""

import os
import time
import torch
import numpy as np
import pandas as pd
import pickle
import logging
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, roc_auc_score, precision_recall_curve, auc)
import argparse

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fraud_detection_train')

def load_pt_file(file_path):
    """
    Carga un archivo .pt con datos preprocesados.
    
    Args:
        file_path (str): Ruta al archivo .pt
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, feature_names)
    """
    logger.info(f"Cargando datos desde: {file_path}")
    try:
        data = torch.load(file_path)
        X_train = data['X_train'].numpy()
        y_train = data['y_train'].numpy()
        X_test = data['X_test'].numpy()
        y_test = data['y_test'].numpy()
        feature_names = data['feature_names']
        logger.info(f"Datos cargados correctamente.")
        logger.info(f"Dimensiones X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"Dimensiones X_test: {X_test.shape}, y_test: {y_test.shape}")
        logger.info(f"Distribución de clases en entrenamiento: {Counter(y_train)}")
        logger.info(f"Distribución de clases en prueba: {Counter(y_test)}")
        
        return X_train, y_train, X_test, y_test, feature_names
    except Exception as e:
        logger.error(f"Error al cargar los datos: {e}")
        raise

def define_models():
    """
    Define los modelos a entrenar.
    
    Returns:
        dict: Diccionario con los modelos
    """
    logger.info("Definiendo modelos...")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,          # Recomendado por Dal Pozzolo et al.
            max_depth=None,            # Permitir árboles completos para capturar patrones complejos
            min_samples_split=2,       # Valor predeterminado efectivo para detección de fraude
            min_samples_leaf=1,        # Valor predeterminado efectivo para detección de fraude
            random_state=42,
            n_jobs=-1                  # Usar todos los núcleos disponibles
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,             # Valor recomendado por Whitrow et al.
            weights='distance',        # Ponderación por distancia, efectiva para fraude (Bahnsen et al.)
            algorithm='auto',          # Selecciona automáticamente el algoritmo óptimo
            leaf_size=30,              # Valor predeterminado
            p=2,                       # Distancia euclidiana
            n_jobs=-1                  # Usar todos los núcleos disponibles
        ),
        'SVM Polinómico': SVC(
            kernel='poly',             # Kernel polinómico de grado 2
            degree=2,                  # Grado del polinomio
            C=1.0,                     # Parámetro de regularización
            gamma='scale',             # Escala de acuerdo con número de características
            coef0=0.0,                 # Término independiente en función kernel
            probability=True,          # Necesario para ROC y PR curves
            random_state=42
        ),
        'SVM Sigmoide': SVC(
            kernel='sigmoid',          # Kernel sigmoide
            C=1.0,                     # Parámetro de regularización
            gamma='scale',             # Escala de acuerdo con número de características
            coef0=0.0,                 # Término independiente en función kernel
            probability=True,          # Necesario para ROC y PR curves
            random_state=42
        ),
        'SVM RBF': SVC(
            kernel='rbf',              # Kernel de función de base radial
            C=1.0,                     # Parámetro de regularización
            gamma='scale',             # Escala de acuerdo con número de características
            probability=True,          # Necesario para ROC y PR curves
            random_state=42
        )
    }
    
    logger.info(f"Se han definido {len(models)} modelos: {', '.join(models.keys())}")
    return models

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Entrena y evalúa un modelo.
    
    Args:
        model: Modelo a evaluar
        X_train: Características de entrenamiento
        y_train: Etiquetas de entrenamiento
        X_test: Características de prueba
        y_test: Etiquetas de prueba
        model_name: Nombre del modelo para registro
        
    Returns:
        tuple: (resultados, modelo_entrenado)
    """
    logger.info(f"Evaluando modelo: {model_name}")
    
    # Medir tiempo de entrenamiento
    start_time = time.time()
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Calcular tiempo de entrenamiento
    train_time = time.time() - start_time
    
    # Medir tiempo de predicción
    start_time = time.time()
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular probabilidades (para ROC-AUC)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Alternativa para modelos que no soportan predict_proba
        y_prob = model.decision_function(X_test) if hasattr(model, "decision_function") else None
    
    # Calcular tiempo de predicción
    pred_time = time.time() - start_time
    
    # Calcular métricas básicas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC-AUC solo si tenemos probabilidades
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    # Calcular PR-AUC (Area bajo la curva Precision-Recall)
    if y_prob is not None:
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
    else:
        pr_auc = None
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Calcular métricas adicionales específicas para detección de fraude
    tn, fp, fn, tp = cm.ravel()
    
    # Tasa de falsas alarmas (False Positive Rate)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Guardar resultados (SIN incluir el modelo aquí)
    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'False Positive Rate': fpr,
        'Train Time (s)': train_time,
        'Prediction Time (s)': pred_time,
        'Confusion Matrix': cm,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'y_test': y_test  # Guardamos también las etiquetas reales
    }
    
    # Mostrar resultados básicos
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}" if roc_auc else "  ROC-AUC: N/A")
    logger.info(f"  PR-AUC: {pr_auc:.4f}" if pr_auc else "  PR-AUC: N/A")
    logger.info(f"  False Positive Rate: {fpr:.4f}")
    logger.info(f"  Tiempo entrenamiento: {train_time:.2f} s")
    logger.info(f"  Tiempo predicción: {pred_time:.2f} s")
    
    # Mostrar matriz de confusión
    logger.info("  Matriz de Confusión:")
    logger.info(f"    TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    logger.info(f"    FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    return results, model  # Devolver resultados Y modelo por separado

def train_models_on_dataset(models, X_train, y_train, X_test, y_test, dataset_name):
    """
    Entrena todos los modelos en un conjunto de datos específico.
    
    Args:
        models (dict): Diccionario con los modelos a entrenar
        X_train: Características de entrenamiento
        y_train: Etiquetas de entrenamiento
        X_test: Características de prueba
        y_test: Etiquetas de prueba
        dataset_name (str): Nombre del conjunto de datos para el registro
        
    Returns:
        dict: Resultados por modelo
    """
    logger.info(f"Entrenando modelos en conjunto de datos: {dataset_name}")
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Entrenando modelo: {model_name}")
        
        try:
            # Crear una nueva instancia del modelo con los mismos parámetros
            model_instance = type(model)(**model.get_params())
            
            # Entrenar y evaluar (ahora devuelve resultados Y modelo)
            evaluation, trained_model = evaluate_model(model_instance, X_train, y_train, X_test, y_test, model_name)
            
            # Guardar resultados y modelo entrenado por separado
            results[model_name] = evaluation
            results[model_name]['trained_model'] = trained_model  # Añadir modelo a los resultados
            
        except Exception as e:
            logger.error(f"Error al entrenar el modelo {model_name}: {e}")
            results[model_name] = {"Error": str(e)}
    
    return results

def train_all_datasets(data_dir, output_dir='results', sampling_methods=None):
    """
    Entrena modelos en todos los conjuntos de datos disponibles.
    
    Args:
        data_dir (str): Directorio con los archivos .pt
        output_dir (str): Directorio para guardar resultados
        sampling_methods (list): Lista de métodos de sampling a procesar. Si es None, se procesan todos.
        
    Returns:
        dict: Resultados para todos los datasets y modelos
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Si no se especifican métodos, usar todos los disponibles
    if sampling_methods is None:
        sampling_methods = ['original', 'smote', 'adasyn', 'smote_tomek']
    
    logger.info(f"Procesando los siguientes métodos de sampling: {', '.join(sampling_methods)}")
    
    # Definir modelos
    models = define_models()
    
    # Almacenar resultados por dataset
    all_results = {}
    
    # Procesar cada método de sampling
    for method in sampling_methods:
        try:
            # Ruta al archivo .pt
            data_path = os.path.join(data_dir, f"{method}.pt")
            
            # Verificar si existe
            if not os.path.exists(data_path):
                logger.warning(f"No se encontró el archivo {data_path}. Omitiendo.")
                continue
            
            # Cargar datos
            X_train, y_train, X_test, y_test, feature_names = load_pt_file(data_path)
            
            # Entrenar modelos
            results = train_models_on_dataset(models, X_train, y_train, X_test, y_test, method)
            
            # Guardar resultados
            all_results[method] = results
            
            # GUARDAR MODELOS Y PREDICCIONES (2 archivos por modelo)
            
            # Crear directorio para modelos específico del método
            models_dir = os.path.join(output_dir, 'models', method)
            os.makedirs(models_dir, exist_ok=True)
            logger.info(f"Directorio de modelos creado/verificado: {os.path.abspath(models_dir)}")
            
            # Guardar modelos entrenados y predicciones en carpetas organizadas
            for model_name, result in results.items():
                if "Error" not in result and 'trained_model' in result:
                    # Limpiar nombre del modelo para usar como nombre de archivo
                    clean_model_name = model_name.replace(' ', '_').replace('-', '_').lower()
                    logger.info(f"Procesando modelo: {model_name} → {clean_model_name}")
                    
                    # ARCHIVO 1: MODELO ENTRENADO (.pkl)
                    model_filename = f"{clean_model_name}.pkl"
                    model_path = os.path.join(models_dir, model_filename)
                    logger.info(f"Guardando MODELO en: {os.path.abspath(model_path)}")
                    
                    try:
                        with open(model_path, 'wb') as f:
                            pickle.dump(result['trained_model'], f)  # Solo el modelo
                        
                        # Verificar que se guardó correctamente
                        if os.path.exists(model_path):
                            file_size = os.path.getsize(model_path)
                            logger.info(f"MODELO {model_name} guardado correctamente")
                            logger.info(f"   Ubicación: {os.path.abspath(model_path)}")
                            logger.info(f"   Tamaño: {file_size} bytes")
                        else:
                            logger.error(f"Error: No se pudo verificar que el modelo se guardó en {model_path}")
                            
                    except Exception as e:
                        logger.error(f"Error al guardar modelo {model_name}: {e}")
                        continue
                    
                    # ARCHIVO 2: PREDICCIONES Y DATOS (_predictions.pkl)
                    predictions_filename = f"{clean_model_name}_predictions.pkl"
                    predictions_path = os.path.join(models_dir, predictions_filename)
                    logger.info(f"Guardando PREDICCIONES en: {os.path.abspath(predictions_path)}")
                    
                    # Datos a guardar en el archivo de predicciones
                    predictions_data = {
                        'y_test': result['y_test'],                    
                        'y_pred': result['y_pred'],                    
                        'y_prob': result['y_prob'],                    
                        'confusion_matrix': result['Confusion Matrix'], 
                        'model_name': model_name,                      
                        'method': method,                              
                        'metrics': {                                   
                            'accuracy': result['Accuracy'],
                            'precision': result['Precision'], 
                            'recall': result['Recall'],
                            'f1_score': result['F1-Score'],
                            'roc_auc': result['ROC-AUC'],
                            'pr_auc': result['PR-AUC'],
                            'false_positive_rate': result['False Positive Rate']
                        }
                    }
                    
                    try:
                        with open(predictions_path, 'wb') as f:
                            pickle.dump(predictions_data, f)  # Predicciones + metadata
                        
                        # Verificar que se guardó correctamente
                        if os.path.exists(predictions_path):
                            file_size = os.path.getsize(predictions_path)
                            logger.info(f"PREDICCIONES de {model_name} guardadas correctamente")
                            logger.info(f"   Ubicación: {os.path.abspath(predictions_path)}")
                            logger.info(f"   Tamaño: {file_size} bytes")
                        else:
                            logger.error(f"Error: No se pudo verificar que las predicciones se guardaron en {predictions_path}")
                            
                    except Exception as e:
                        logger.error(f"Error al guardar predicciones de {model_name}: {e}")
                        
                else:
                    if "Error" in result:
                        logger.warning(f"No se guardaron archivos para {model_name} debido a errores en el entrenamiento")
                    else:
                        logger.warning(f"No se encontró modelo entrenado para {model_name}")
            

            # GUARDAR TABLA CSV DE RESULTADOS
            results_df = create_results_df(results)
            results_path = os.path.join(output_dir, f"results_{method}.csv")
            results_df.to_csv(results_path, index=True)
            logger.info(f"Resultados CSV para {method} guardados en: {os.path.abspath(results_path)}")
            
        except Exception as e:
            logger.error(f"Error procesando el método {method}: {e}")
    
    return all_results

def create_results_df(results_dict):
    """
    Crea un DataFrame a partir de un diccionario de resultados.
    
    Args:
        results_dict (dict): Diccionario con resultados por modelo
        
    Returns:
        pd.DataFrame: DataFrame con los resultados
    """
    data = []
    
    for model_name, results in results_dict.items():
        # Si hubo un error, incluir solo ese campo
        if "Error" in results:
            row = {'Modelo': model_name, 'Error': results["Error"]}
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 
                          'False Positive Rate', 'Train Time (s)', 'Prediction Time (s)']:
                row[metric] = None
        else:
            # Crear fila con todas las métricas
            row = {
                'Modelo': model_name,
                'Accuracy': results['Accuracy'],
                'Precision': results['Precision'],
                'Recall': results['Recall'],
                'F1-Score': results['F1-Score'],
                'ROC-AUC': results['ROC-AUC'],
                'PR-AUC': results['PR-AUC'],
                'False Positive Rate': results['False Positive Rate'],
                'Train Time (s)': results['Train Time (s)'],
                'Prediction Time (s)': results['Prediction Time (s)']
            }
        
        data.append(row)
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    if not df.empty and 'Modelo' in df.columns:
        df.set_index('Modelo', inplace=True)
    
    return df

if __name__ == "__main__":
    # Si se ejecuta como script principal
    parser = argparse.ArgumentParser(description='Entrenamiento para detección de fraude')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directorio con los archivos .pt preprocesados')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directorio donde guardar los resultados')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['original', 'smote', 'adasyn', 'smote_tomek'],
                        help='Métodos de sampling a procesar')
    
    args = parser.parse_args()
    
    # Ejecutar entrenamiento
    train_all_datasets(
        args.data_dir,
        args.output_dir,
        args.methods
    )