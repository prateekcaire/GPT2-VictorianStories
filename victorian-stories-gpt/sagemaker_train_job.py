
import sagemaker
from sagemaker.pytorch import PyTorch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Setting up SageMaker session")
try:
    sagemaker_session = sagemaker.Session()
    logger.info("SageMaker session created successfully")
except Exception as e:
    logger.error(f"Failed to create SageMaker session: {e}")
    raise

role = "arn:aws:iam::961944346110:role/SageMakerExecutionRole"

# Define the S3 bucket and path for saving the model
custom_prefix = "victorian-llm-"
output_path = f's3://prateekmodels/gpt-2/output/{custom_prefix}'
train_data_uri = 's3://prateekmodels/data/train/gpt-2/pg-19/'

# Create the PyTorch estimator
logger.info("Creating the PyTorch estimator")
try:
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='.',
        role=role,
        framework_version='2.0.1',
        py_version='py310',
        instance_count=1,
        instance_type='ml.p4d.24xlarge',
        volume_size=30,
        max_run=72000,
        output_path=output_path,
        environment={
            'NCCL_P2P_DISABLE': '1',
            'NCCL_DEBUG': 'INFO',
            'PYTHONFAULTHANDLER': '1',
            'TORCH_DISTRIBUTED_DEBUG': 'DETAIL'
        },
        hyperparameters={
            'n_layers': 12,
            'n_channels': 768,
            'n_vocab': 50304,
            'n_tokens': 2048,
            'n_heads': 12,
            'batch_size': 8,
            'lr': 1e-4,
            'epochs': 5,
            'dropout': 0.2
        },
        distribution={
            'torch_distributed': {
                'enabled': True,
                'process_group_backend': 'nccl'
            }
        },
        sagemaker_session=sagemaker_session,
        dependencies=['train_requirements.txt'],
        disable_profiler=True
    )
    logger.info("PyTorch estimator created successfully")
except Exception as e:
    logger.error(f"Failed to create PyTorch estimator: {e}")
    raise

# Start the training job
if __name__ == '__main__':
    try:
        logger.info("Starting the training job")
        estimator.fit({'training': train_data_uri})
        logger.info("Training job started successfully")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
