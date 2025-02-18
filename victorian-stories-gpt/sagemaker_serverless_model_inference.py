import json
import torch
from model import LanguageModel
import os
import tiktoken
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def model_fn(model_dir):
    logger.info(f"Loading model from {model_dir}")
    model_path = os.path.join(model_dir, "model.pth")
    try:
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        args = model_dict['args']
        model = LanguageModel(
            n_layers=args['n_layers'],
            n_channels=args['n_channels'],
            n_vocab=args['n_vocab'],
            n_tokens=args['n_tokens'],
            n_heads=args['n_heads'],
            f_dropout=args['dropout']
        )

        # Remove unexpected prefixes from state dict
        state_dict = model_dict['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module._orig_mod.'):
                new_k = k.replace('module._orig_mod.', '')
            elif k.startswith('module.'):
                new_k = k.replace('module.', '')
            else:
                new_k = k
            new_state_dict[new_k] = v

        # Load the cleaned state dict
        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def input_fn(request_body, request_content_type):
    logger.info(f"Received request with content type: {request_content_type}")
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        logger.info(f"Parsed input: {input_data}")
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    logger.info("Starting prediction")
    prompt = input_data['prompt']
    max_tokens = input_data.get('max_tokens', 100)

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_prompt = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

    generated_text = model.generate(
        encoded_prompt,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
        temperature=1.3,  # You can adjust these parameters
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )

    logger.info(f"Generated text: {generated_text}")
    return generated_text


def output_fn(prediction_output, accept):
    logger.info(f"Formatting output with accept type: {accept}")
    if accept == 'application/json':
        response = json.dumps({"generated_text": prediction_output})
        logger.info(f"Formatted response: {response}")
        return response, accept
    raise ValueError(f"Unsupported accept type: {accept}")


# Local testing
if __name__ == "__main__":
    # Specify the path to your local model directory
    local_model_dir = "./model"

    # Load the model
    model = model_fn(local_model_dir)

    # Test input
    test_input = json.dumps({
        "prompt": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want ",
        "max_tokens": 50
    })

    # Process input
    processed_input = input_fn(test_input, 'application/json')

    # Generate prediction
    prediction = predict_fn(processed_input, model)

    # Format output
    output, _ = output_fn(prediction, 'application/json')

    print("Local test result:")
    print(output)