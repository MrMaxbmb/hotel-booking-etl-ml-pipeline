import sys
import os
from datetime import datetime

# Добавляем корневую папку проекта в PYTHONPATH
# Получаем путь к DAG файлу и поднимаемся на 2 уровня вверх
dag_dir = os.path.dirname(os.path.abspath(__file__))  # airflow/dags/
airflow_dir = os.path.dirname(dag_dir)  # airflow/
project_root = os.path.dirname(airflow_dir)  # project root
print(f"DEBUG: Adding to PYTHONPATH: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.steps.preprocess_step import PreprocessStep
from src.steps.inference_step import InferenceStep
from src.steps.feature_engineering_step import FeatureEngineeringStep
from src.steps.utils.data_classes import PreprocessingData, FeaturesEngineeringData
from src.steps.config import (
    FeatureEngineeringConfig,
    INFERENCE_DATA_PATH,
    PreprocessConfig,
)


# True означает, что это pipeline для предсказаний
inference_mode = True
preprocessing_data = PreprocessingData(
    batch_path=PreprocessConfig.batch_path
)
features_engineering_data = FeaturesEngineeringData(
    batch_path=FeatureEngineeringConfig.batch_path,
    encoders_path=FeatureEngineeringConfig.encoders_path,
)

# По аналогии
preprocess_step = PreprocessStep(
    inference_mode=inference_mode, 
    preprocessing_data=preprocessing_data
)
feature_engineering_step = FeatureEngineeringStep(
    inference_mode=inference_mode,
    feature_engineering_data=features_engineering_data
)
inference_step = InferenceStep()


default_args = {
    "owner": "user",
    "depends_on_past": False,
    "retries": 0,
    "catchup": False,
}

with DAG(
    "inference-pipeline",
    default_args=default_args,
    start_date=datetime(2025, 9, 20),
    tags=["inference"],
    schedule=None,
) as dag:
    preprocessing_task = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocess_step,
        op_kwargs={
            "data_path": INFERENCE_DATA_PATH
        }
    )
    feature_engineering_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering_step,
        op_kwargs={
            "batch_path": preprocessing_data.batch_path
        }
    )
    inference_task = PythonOperator(
        task_id="inference",
        python_callable=inference_step,
        op_kwargs={
            "batch_path": features_engineering_data.batch_path
        }
    )

    preprocessing_task >> feature_engineering_task >> inference_task
