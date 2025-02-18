import streamlit as st
import torch
import tiktoken
import sys
import os
from model import LanguageModel
import time
import subprocess


@st.cache_resource
def load_model(model_path):
    """Load the model and return it (cached by Streamlit)"""
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            st.info("Please check if the model file exists in the correct location")
            return None, None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load the saved model
        model_dict = torch.load(model_path, map_location=device)
        args = model_dict['args']

        # Create model with saved parameters
        model = LanguageModel(
            n_layers=args['n_layers'],
            n_channels=args['n_channels'],
            n_vocab=args['n_vocab'],
            n_tokens=args['n_tokens'],
            n_heads=args['n_heads'],
            f_dropout=args['dropout'],
            device=device
        )

        # Clean state dict keys
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

        model.load_state_dict(new_state_dict)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def stream_tokens(text_placeholder, full_text=""):
    """Stream tokens to the Streamlit app with blinking cursor effect"""
    time.sleep(0.05)  # Add slight delay for visual effect
    text_placeholder.markdown(f"{full_text}▌")
    return full_text


def generate_with_streaming(model, input_ids, max_tokens, tokenizer, temperature, top_k, top_p, progress_bar,
                          text_placeholder):
    """Generate text with streaming output"""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    generated = input_ids.clone()
    past_tokens = set()
    full_text = ""
    tokens_generated = 0

    with torch.no_grad():
        for _ in range(max_tokens):
            # Update progress
            progress_bar.progress(min(tokens_generated / max_tokens, 1.0))

            # Get predictions
            idx_cond = generated[:, -model.n_tokens:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]

            # Apply temperature and sampling
            logits = logits / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Top-k sampling
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices,
                                                                                      sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample token
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat((generated, next_token), dim=1)

            # Decode and display new token
            new_token = tokenizer.decode([next_token.item()])
            full_text += new_token
            stream_tokens(text_placeholder, full_text)

            tokens_generated += 1

            # Stop if end token is generated
            if next_token.item() == tokenizer.eot_token:
                break

    return full_text


def main():
    st.set_page_config(
        page_title="Victorian Literature Generator",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Victorian Literature Text Generator")
    st.markdown("Generate text in the style of Victorian literature using a GPT-2 style model.")

    # Load model with absolute path
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    VICTORIAN_STORIES_DIR = os.path.dirname(CURRENT_DIR)  # victorian-stories-gpt directory
    model_path = os.path.join(CURRENT_DIR, "downloaded_models", "extracted", "model.pth")
    st.info(f"Loading model from: {model_path}")
    model, device = load_model(model_path)

    if model is None:
        st.error("Failed to load model. Please check the model path and try again.")
        return

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create two columns for parameters
    col1, col2 = st.columns(2)

    with col1:
        max_tokens = st.slider("Maximum tokens to generate", 10, 500, 100)
        temperature = st.slider("Temperature", 0.1, 2.0, 1.3)

    with col2:
        top_k = st.slider("Top K", 0, 100, 50)
        top_p = st.slider("Top P", 0.0, 1.0, 0.95)

    # Text input
    prompt = st.text_area(
        "Enter your prompt:",
        height=100,
        placeholder="Enter the beginning of your Victorian story here..."
    )

    if st.button("Generate", type="primary"):
        if not prompt:
            st.warning("Please enter a prompt first.")
            return

        try:
            with st.spinner("Generating text..."):
                # Create placeholder for streaming text
                text_placeholder = st.empty()
                progress_bar = st.progress(0)

                # Tokenize input
                input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

                # Generate text with streaming
                generated_text = generate_with_streaming(
                    model=model,
                    input_ids=input_ids,
                    max_tokens=max_tokens,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    progress_bar=progress_bar,
                    text_placeholder=text_placeholder
                )

                # Display final text and stats
                st.markdown("### Generated Text:")
                st.markdown(generated_text)

                # Display generation stats
                st.markdown("---")
                st.markdown("### Generation Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Input Length", len(prompt.split()))
                with col2:
                    st.metric("Generated Length", len(generated_text.split()) - len(prompt.split()))
                with col3:
                    st.metric("Total Length", len(generated_text.split()))

        except Exception as e:
            st.error(f"An error occurred during generation: {str(e)}")


if __name__ == "__main__":
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change to the script directory
    os.chdir(script_dir)

    # Run streamlit command
    cmd = [sys.executable, '-m', 'streamlit', 'run', __file__,
           '--server.port=8501', '--server.address=localhost']

    if not os.environ.get('STREAMLIT_RUN_MODE'):
        os.environ['STREAMLIT_RUN_MODE'] = 'true'
        subprocess.run(cmd)
    else:
        main()