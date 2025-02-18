FROM public.ecr.aws/lambda/python:3.8

# Install system dependencies
RUN yum install -y gcc-c++ make

# Copy function code and model files
COPY lambda_model_inference.py ${LAMBDA_TASK_ROOT}
COPY model.py ${LAMBDA_TASK_ROOT}

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Explicitly install tiktoken and verify installation
RUN pip install --no-cache-dir tiktoken && \
    python -c "import tiktoken; print('tiktoken successfully imported')"

# List all installed packages for debugging
RUN pip list

# Set the CMD to your handler
CMD [ "lambda_model_inference.lambda_handler" ]