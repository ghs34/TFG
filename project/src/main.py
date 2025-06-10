#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para el proyecto de detección de fraude en transacciones financieras.

Este script coordina todo el flujo de trabajo del proyecto: desde el preprocesamiento
de los datos hasta el entrenamiento de modelos y la evaluación de resultados.

Autor: Hlieb Sydorenko
Fecha: 10/06/2025
TFG: Comparativa de Modelos de Aprendizaje Automático y Estrategias de Aumentación de Datos para la Detección de Fraude en Transacciones Financieras
"""

import os
import argparse
import logging
import time
import sys
from datetime import datetime

# Intentar importar módulos propios
try:
    # Añadir directorio actual al path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    import preprocess
    import train
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que preprocess.py y train.py están en el mismo directorio o en PYTHONPATH")
    sys.exit(1)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fraud_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('fraud_detection_main')

def create_project_structure(base_dir):
    """
    Crea la estructura de directorios del proyecto.
    
    Args:
        base_dir (str): Directorio base del proyecto
    """
    # Directorios necesarios
    dirs = [
        os.path.join(base_dir, 'data'),
        os.path.join(base_dir, 'data/raw'),
        os.path.join(base_dir, 'data/processed'),
        os.path.join(base_dir, 'results'),
        os.path.join(base_dir, 'results/models'),
        os.path.join(base_dir, 'notebooks')
    ]
    
    # Crear directorios
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directorio creado (si no existía): {directory}")

def run_full_pipeline(data_path, base_dir='.', test_size=0.2, random_state=42, sampling_methods=None):
    """
    Ejecuta todo el flujo de trabajo del proyecto.
    
    Args:
        data_path (str): Ruta al archivo CSV con los datos
        base_dir (str): Directorio base del proyecto
        test_size (float): Proporción para el conjunto de prueba
        random_state (int): Semilla para reproducibilidad
        sampling_methods (list): Lista de métodos de sampling a procesar
        
    Returns:
        dict: Resultados del pipeline
    """
    # Registro de tiempo de inicio
    start_time = time.time()
    logger.info(f"Iniciando pipeline completo. Fecha/hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. Crear estructura de directorios
        logger.info("Creando estructura de directorios...")
        create_project_structure(base_dir)
        
        # 2. Rutas a directorios
        processed_dir = os.path.join(base_dir, 'data/processed')
        results_dir = os.path.join(base_dir, 'results')
        
        # 3. Ejecutar preprocesamiento
        logger.info("Iniciando preprocesamiento...")
        output_files = preprocess.preprocess(
            data_path,
            output_dir=processed_dir,
            test_size=test_size,
            random_state=random_state
        )
        logger.info(f"Preprocesamiento completado. Archivos generados: {list(output_files.keys())}")
        
        # 4. Entrenar modelos
        logger.info("Iniciando entrenamiento de modelos...")
        all_results = train.train_all_datasets(
            processed_dir,
            output_dir=results_dir,
            sampling_methods=sampling_methods
        )
        logger.info(f"Entrenamiento completado. Métodos procesados: {list(all_results.keys())}")
        
        # Información adicional sobre los archivos generados
        logger.info("Archivos generados:")
        logger.info(f"  - Resultados por método: results_[método].csv")
        logger.info(f"  - Modelos entrenados: {os.path.join(results_dir, 'models/')}")
        
        # 5. Registrar tiempo total
        total_time = time.time() - start_time
        logger.info(f"Pipeline completado en {total_time:.2f} segundos")
        logger.info(f"Los resultados se han guardado en: {results_dir}")
        
        return {
            'status': 'success',
            'output_files': output_files,
            'results': all_results,
            'execution_time': total_time
        }
        
    except Exception as e:
        logger.error(f"Error en la ejecución del pipeline: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }

if __name__ == "__main__":
    # Si se ejecuta como script principal
    parser = argparse.ArgumentParser(description='Pipeline completo para detección de fraude')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Ruta al archivo CSV con los datos')
    parser.add_argument('--base_dir', type=str, default='.',
                        help='Directorio base del proyecto')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proporción para el conjunto de prueba')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Semilla para reproducibilidad')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['original', 'smote', 'adasyn', 'smote_tomek'],
                        help='Métodos de sampling a procesar')
    
    args = parser.parse_args()
    
    # Validar que el archivo de datos existe
    if not os.path.exists(args.data_path):
        print(f"Error: El archivo {args.data_path} no existe.")
        sys.exit(1)
    
    # Ejecutar pipeline completo
    result = run_full_pipeline(
        args.data_path,
        args.base_dir,
        args.test_size,
        args.random_state,
        args.methods
    )
    
    # Código de salida
    if result['status'] == 'success':
        sys.exit(0)
    else:
        sys.exit(1)