# mlflow_artifact_loader.py
from mlflow import artifacts
from mlflow.exceptions import MlflowException
import joblib
import json
import yaml
import pickle
from typing import Any, Dict, Union, Optional
import tempfile

class MLflowArtifactLoader:
    """
    仿 mlflow.artifacts.load_dict 的通用 artifact 加载工具类，
    支持 joblib、json、yaml、pickle 等格式。
    """

    @staticmethod
    def load_joblib(artifact_uri: str, tracking_uri: Optional[str] = None) -> Any:
        """
        从远程 URI 加载 joblib 保存的文件（.pkl, .joblib）

        Args:
            artifact_uri: 远程 artifact URI，如：
                - models:/MyModel/Production/encoder.pkl
                - runs:/abc123/artifacts/models/scaler.pkl
                - s3://bucket/path/to/file.pkl
            tracking_uri: MLflow Tracking Server 地址（可选）

        Returns:
            joblib.load() 加载的对象

        Example:
            encoder = MLflowArtifactLoader.load_joblib(
                "models:/HousingPriceModel@Production/models/ocean_encoder.pkl"
            )
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = artifacts.download_artifacts(
                artifact_uri=artifact_uri,
                dst_path=tmpdir,
                tracking_uri=tracking_uri
            )
            try:
                return joblib.load(local_path)
            except Exception as e:
                raise MlflowException(f"Failed to load joblib artifact from {artifact_uri}: {e}", error_code="BAD_REQUEST")

    @staticmethod
    def load_pickle(artifact_uri: str, tracking_uri: Optional[str] = None) -> Any:
        """
        加载标准 pickle 文件（.pkl）
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = artifacts.download_artifacts(
                artifact_uri=artifact_uri,
                dst_path=tmpdir,
                tracking_uri=tracking_uri
            )
            with open(local_path, 'rb') as f:
                try:
                    return pickle.load(f)
                except Exception as e:
                    raise MlflowException(f"Failed to unpickle artifact from {artifact_uri}: {e}", error_code="BAD_REQUEST")

    @staticmethod
    def load_json(artifact_uri: str, tracking_uri: Optional[str] = None) -> Dict[str, Any]:
        """
        加载 JSON 文件（.json）
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = artifacts.download_artifacts(
                artifact_uri=artifact_uri,
                dst_path=tmpdir,
                tracking_uri=tracking_uri
            )
            with open(local_path, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError as e:
                    raise MlflowException(f"Invalid JSON in {artifact_uri}: {e}", error_code="BAD_REQUEST")

    @staticmethod
    def load_yaml(artifact_uri: str, tracking_uri: Optional[str] = None) -> Dict[str, Any]:
        """
        加载 YAML 文件（.yml, .yaml）
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = artifacts.download_artifacts(
                artifact_uri=artifact_uri,
                dst_path=tmpdir,
                tracking_uri=tracking_uri
            )
            with open(local_path, 'r', encoding='utf-8') as f:
                try:
                    return yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise MlflowException(f"Invalid YAML in {artifact_uri}: {e}", error_code="BAD_REQUEST")

    @staticmethod
    def load_text(artifact_uri: str, tracking_uri: Optional[str] = None) -> str:
        """
        加载纯文本文件（.txt, .log, .md 等）
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = artifacts.download_artifacts(
                artifact_uri=artifact_uri,
                dst_path=tmpdir,
                tracking_uri=tracking_uri
            )
            with open(local_path, 'r', encoding='utf-8') as f:
                return f.read()

    @staticmethod
    def load_bytes(artifact_uri: str, tracking_uri: Optional[str] = None) -> bytes:
        """
        加载二进制文件（如图片、PDF 等）
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = artifacts.download_artifacts(
                artifact_uri=artifact_uri,
                dst_path=tmpdir,
                tracking_uri=tracking_uri
            )
            with open(local_path, 'rb') as f:
                return f.read()

    @staticmethod
    def download_to_path(artifact_uri: str, dst_path: str, tracking_uri: Optional[str] = None) -> str:
        """
        将 artifact 下载到指定路径（不加载），适合大文件或多次使用

        Returns:
            下载后的本地路径
        """
        return artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            dst_path=dst_path,
            tracking_uri=tracking_uri
        )