import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deploy_standard_endpoint(
    model_data,
    role,
    source_dir,
    entry_point="sagemaker_serverless_model_inference.py",
    framework_version="2.0.1",
    py_version="py310",
    endpoint_name="stem-gpt2-serverless-endpoint",
    instance_type="ml.m5.2xlarge",
    instance_count=1
):
    sagemaker_session = sagemaker.Session()
    sagemaker_client = boto3.client('sagemaker')

    logger.info(f"Default SageMaker S3 bucket: {sagemaker_session.default_bucket()}")

    try:
        logger.info(f"Attempting to delete existing endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Waiting for endpoint {endpoint_name} to be deleted...")
        waiter = sagemaker_client.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=endpoint_name)
    except sagemaker_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            logger.info(f"Endpoint {endpoint_name} does not exist. Proceeding with creation.")
        else:
            raise

    try:
        logger.info(f"Attempting to delete existing endpoint configuration: {endpoint_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        time.sleep(30)  # Wait for the deletion to complete
    except sagemaker_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            logger.info(f"Endpoint configuration {endpoint_name} does not exist. Proceeding with creation.")
        else:
            raise

    model = PyTorchModel(
        model_data=model_data,
        role=role,
        entry_point=entry_point,
        source_dir=source_dir,
        framework_version=framework_version,
        py_version=py_version,
        code_location=f"s3://{sagemaker_session.default_bucket()}/stem-gpt2-model-code",
        dependencies=['requirements.txt'],
        env={
            'PYTHONUNBUFFERED': '1'
        }
    )



    try:
        # Deploy the model to a standard endpoint
        predictor = model.deploy(
            endpoint_name=endpoint_name,
            instance_type=instance_type,
            initial_instance_count=instance_count
        )
    except Exception as e:
        logger.error(f"Error deploying endpoint: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    return predictor


if __name__ == "__main__":
    try:
        model_data = "s3://prateekmodels/gpt-2/output/victorian-llm-/pytorch-training-2024-10-03-17-18-53-694/output/model.tar.gz"
        role = "arn:aws:iam::961944346110:role/SageMakerExecutionRole"
        source_dir = "./"

        logger.info(f"Model data: {model_data}")
        logger.info(f"IAM role: {role}")
        logger.info(f"Source directory: {source_dir}")

        predictor = deploy_standard_endpoint(model_data, role, source_dir)
        logger.info(f"Endpoint deployed: {predictor.endpoint_name}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise