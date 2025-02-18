import torch
import boto3
import io
import tarfile
import os

session = boto3.Session(profile_name='prateek_personal')
s3 = session.client('s3')

BUCKET_NAME = 'prateekmodels'
MODEL_PATH = 'gpt-2/output/victorian-llm-/pytorch-training-2024-10-03-17-18-53-694/output/model.tar.gz'
# MODEL_PATH = 'gpt-2/output/pytorch-training-2024-10-03-17-18-53-694/output/model.tar.gz'
LOCAL_SAVE_DIR = 'downloaded_models'


def inspect_saved_model():
    try:
        # Verify AWS credentials
        try:
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            print(f"Using AWS Account: {identity['Account']}")
        except Exception as e:
            print(f"Failed to get AWS identity: {str(e)}")
            return

        # Create local directory if it doesn't exist
        os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
        local_path = os.path.join(LOCAL_SAVE_DIR, 'model.tar.gz')

        print(f"\nAttempting to download and inspect model from s3://{BUCKET_NAME}/{MODEL_PATH}")
        print(f"Downloading model to {local_path}")

        # Download the tar.gz file
        s3.download_file(BUCKET_NAME, MODEL_PATH, local_path)
        print("Download complete!")

        # Extract the tar.gz file
        print("\nExtracting contents...")
        with tarfile.open(local_path, 'r:gz') as tar:
            extract_path = os.path.join(LOCAL_SAVE_DIR, 'extracted')
            os.makedirs(extract_path, exist_ok=True)

            # List contents before extraction
            contents = tar.getnames()
            print(f"Contents of tar.gz file: {contents}")

            # Extract all files
            tar.extractall(path=extract_path)
            print(f"Files extracted to: {extract_path}")

            # Find the .pth file
            model_file_name = next((name for name in contents if name.endswith('.pth')), None)
            if model_file_name is None:
                raise ValueError("No .pth file found in tar.gz archive")

            model_path = os.path.join(extract_path, model_file_name)
            print(f"\nFound model file: {model_path}")

            # Load and inspect the model
            print("Loading model for inspection...")
            model_dict = torch.load(model_path, map_location='cpu')

        # Inspect the contents
        print("\nKeys in the model_dict:")
        for key in model_dict.keys():
            print(f"- {key}")

        if 'model_state_dict' in model_dict:
            state_dict = model_dict['model_state_dict']
            print("\nKeys in the model_state_dict:")
            for key in state_dict.keys():
                print(f"- {key}")
                if hasattr(state_dict[key], 'shape'):
                    print(f"  Shape: {state_dict[key].shape}")
        else:
            print("Error: 'model_state_dict' not found in the saved model.")

        if 'args' in model_dict:
            print("\nSaved arguments:")
            for key, value in model_dict['args'].items():
                print(f"- {key}: {value}")
        else:
            print("Error: 'args' not found in the saved model.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    inspect_saved_model()