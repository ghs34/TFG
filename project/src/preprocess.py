#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de preprocesamiento para detección de fraude en transacciones financieras.

Este script realiza el preprocesamiento completo del dataset de fraude en tarjetas de crédito,
incluyendo normalización, ingeniería de características y aplicación de técnicas de balanceo
de clases (SMOTE, ADASYN, SMOTETomek). Guarda los resultados en archivos .pt para uso posterior.

Autor: Hlieb Sydorenko
Fecha: 10/06/2025
TFG: Comparativa de Modelos de Aprendizaje Automático y Estrategias de Aumentación de Datos para la Detección de Fraude en Transacciones Financieras
"""

import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from collections import Counter
import logging
import argparse

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fraud_detection_preprocess')

def load_data(filepath):
    """
    Carga el dataset desde un archivo CSV.
    
    Args:
        filepath (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    logger.info(f"Cargando datos desde: {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Datos cargados correctamente. Dimensiones: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error al cargar los datos: {e}")
        raise

def check_data_quality(df):
    """
    Verifica la calidad de los datos, buscando valores nulos o problemas.
    
    Args:
        df (pd.DataFrame): DataFrame a verificar
        
    Returns:
        bool: True si no hay problemas, False en caso contrario
    """
    # Verificar valores nulos
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning(f"Se encontraron {total_nulls} valores nulos en total")
        logger.warning(f"Columnas con valores nulos: {null_counts[null_counts > 0]}")
        return False
    
    # Verificar distribución de clases
    if 'Class' in df.columns:
        fraud_count = df['Class'].sum()
        total_count = len(df)
        fraud_percentage = (fraud_count / total_count) * 100
        
        logger.info(f"Total de transacciones: {total_count}")
        logger.info(f"Transacciones fraudulentas: {fraud_count} ({fraud_percentage:.4f}%)")
        logger.info(f"Ratio de desbalance: 1:{(total_count - fraud_count) / fraud_count:.2f}")
    
    logger.info("Verificación de calidad completada")
    return True

def create_temporal_features(df):
    """
    Crea características basadas en información temporal.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        
    Returns:
        pd.DataFrame: DataFrame con las nuevas características
    """
    logger.info("Creando características temporales...")
    
    # Crear copia para no modificar el original
    df_enhanced = df.copy()
    
    # Convertir Time a horas
    df_enhanced['Time_Hour'] = df['Time'] / 3600
    
    # Extraer hora del día (0-23)
    df_enhanced['Hour_of_Day'] = df_enhanced['Time_Hour'] % 24
    df_enhanced['Hour_of_Day_Int'] = df_enhanced['Hour_of_Day'].astype(int)
    
    # Características cíclicas para hora del día
    df_enhanced['Hour_sin'] = np.sin(2 * np.pi * df_enhanced['Hour_of_Day'] / 24)
    df_enhanced['Hour_cos'] = np.cos(2 * np.pi * df_enhanced['Hour_of_Day'] / 24)
    
    logger.info("Características temporales creadas")
    return df_enhanced

def create_amount_features(df):
    """
    Crea características basadas en la variable Amount.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        
    Returns:
        pd.DataFrame: DataFrame con las nuevas características
    """
    logger.info("Creando características basadas en Amount...")
    
    # Crear copia para no modificar el original
    df_enhanced = df.copy()
    
    # Transformación logarítmica
    df_enhanced['Amount_Log'] = np.log1p(df['Amount'])
    
    logger.info("Características de Amount creadas")
    return df_enhanced

def split_data(df, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        test_size (float): Proporción para el conjunto de prueba
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Dividiendo datos con test_size={test_size}")
    
    # Separar características y target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Dividir de forma estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"División completada. X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"Distribución en train: {Counter(y_train)}")
    logger.info(f"Distribución en test: {Counter(y_test)}")
    
    return X_train, X_test, y_train, y_test

def normalize_features(X_train, X_test):
    """
    Normaliza las características Time y Amount.
    
    Args:
        X_train (pd.DataFrame): Características de entrenamiento
        X_test (pd.DataFrame): Características de prueba
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    logger.info("Normalizando características Time y Amount...")
    
    # Crear copias para no modificar los originales
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Usar RobustScaler para ser resistente a outliers
    scaler = RobustScaler()
    
    # Aplicar solo a Time y Amount (V1-V28 ya están normalizados)
    columns_to_scale = ['Time', 'Amount']
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    logger.info("Normalización completada")
    return X_train_scaled, X_test_scaled, scaler

def apply_sampling_techniques(X_train, y_train, random_state=42):
    """
    Aplica diferentes técnicas de balanceo de clases.
    
    Args:
        X_train (pd.DataFrame): Características de entrenamiento
        y_train (pd.Series): Etiquetas de entrenamiento
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        dict: Diccionario con los diferentes conjuntos balanceados
    """
    logger.info("Aplicando técnicas de balanceo de clases...")
    
    # SMOTE
    logger.info("Aplicando SMOTE...")
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE completado. Nuevas dimensiones: {X_train_smote.shape}")
    logger.info(f"Distribución después de SMOTE: {Counter(y_train_smote)}")
    
    # ADASYN
    logger.info("Aplicando ADASYN...")
    adasyn = ADASYN(random_state=random_state)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
    logger.info(f"ADASYN completado. Nuevas dimensiones: {X_train_adasyn.shape}")
    logger.info(f"Distribución después de ADASYN: {Counter(y_train_adasyn)}")
    
    # SMOTETomek
    logger.info("Aplicando SMOTETomek...")
    smote_tomek = SMOTETomek(random_state=random_state)
    X_train_smote_tomek, y_train_smote_tomek = smote_tomek.fit_resample(X_train, y_train)
    logger.info(f"SMOTETomek completado. Nuevas dimensiones: {X_train_smote_tomek.shape}")
    logger.info(f"Distribución después de SMOTETomek: {Counter(y_train_smote_tomek)}")
    
    # Crear diccionario con los resultados
    return {
        'original': (X_train, y_train),
        'smote': (X_train_smote, y_train_smote),
        'adasyn': (X_train_adasyn, y_train_adasyn),
        'smote_tomek': (X_train_smote_tomek, y_train_smote_tomek)
    }

def save_as_pt(X_train, y_train, X_test, y_test, output_path):
    """
    Guarda los conjuntos de datos como archivos .pt (PyTorch).
    
    Args:
        X_train (pd.DataFrame): Características de entrenamiento
        y_train (pd.Series): Etiquetas de entrenamiento
        X_test (pd.DataFrame): Características de prueba
        y_test (pd.Series): Etiquetas de prueba
        output_path (str): Ruta donde guardar el archivo
    """
    # Convertir a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar como archivo .pt
    torch.save({
        'X_train': X_train_tensor,
        'y_train': y_train_tensor,
        'X_test': X_test_tensor,
        'y_test': y_test_tensor,
        'feature_names': list(X_train.columns)
    }, output_path)
    
    logger.info(f"Datos guardados en: {output_path}")

def preprocess(data_path, output_dir='data_processed', test_size=0.2, random_state=42):
    """
    Realiza todo el flujo de preprocesamiento y guarda los resultados.
    
    Args:
        data_path (str): Ruta al archivo CSV
        output_dir (str): Directorio donde guardar los archivos .pt
        test_size (float): Proporción para el conjunto de prueba
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        dict: Diccionario con referencias a los archivos guardados
    """
    logger.info("Iniciando preprocesamiento completo...")
    
    # 1. Cargar datos
    df = load_data(data_path)
    
    # 2. Verificar calidad de los datos
    check_data_quality(df)
    
    # 3. Crear características temporales
    df = create_temporal_features(df)
    
    # 4. Crear características basadas en Amount
    df = create_amount_features(df)
    
    # 5. Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size, random_state=random_state)
    
    # 6. Normalizar características
    X_train_scaled, X_test_scaled, _ = normalize_features(X_train, X_test)
    
    # 7. Aplicar técnicas de balanceo de clases
    sampling_results = apply_sampling_techniques(X_train_scaled, y_train, random_state=random_state)
    
    # 8. Guardar cada conjunto en un archivo .pt
    output_files = {}
    for name, (X_sampled, y_sampled) in sampling_results.items():
        output_path = os.path.join(output_dir, f"{name}.pt")
        save_as_pt(X_sampled, y_sampled, X_test_scaled, y_test, output_path)
        output_files[name] = output_path
    
    logger.info("Preprocesamiento completado exitosamente")
    return output_files

if __name__ == "__main__":
    # Si se ejecuta como script principal
    parser = argparse.ArgumentParser(description='Preprocesamiento para detección de fraude')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Ruta al archivo CSV con los datos')
    parser.add_argument('--output_dir', type=str, default='data_processed',
                        help='Directorio donde guardar los archivos .pt')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proporción para el conjunto de prueba')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Semilla para reproducibilidad')
    
    args = parser.parse_args()
    
    # Ejecutar preprocesamiento
    preprocess(
        args.data_path,
        args.output_dir,
        args.test_size,
        args.random_state
    )