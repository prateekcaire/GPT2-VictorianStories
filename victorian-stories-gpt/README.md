# Victorian Literature Generator

## 1. Brief Overview
Implementation of LLM from scratch without using any foundational models. The Victorian Literature Generator is a sophisticated natural language processing system designed to generate text in the style of Victorian literature. It uses a custom GPT-2 style architecture trained on a curated dataset of Victorian-era texts(all the books in Project Gutenberg released before year 1919). The system includes both training and inference components, with deployment options ranging from local Streamlit applications to serverless AWS deployments.

### Key Features
- 
- Custom transformer-based language model
- Multi-platform deployment (local, AWS SageMaker, serverless)
- Interactive text generation with adjustable parameters
- Streamlit-based web interface
- WebSocket support for real-time text streaming
- Distributed training capabilities

## 2. Model Architecture and Parameters

### Core Architecture
- **Base Architecture**: GPT-2 style transformer model
- **Implementation**: PyTorch-based custom implementation
- **Key Components**:
  - CausalSelfAttention
  - MultiAttentionHead
  - FeedForwardNetwork
  - LayerNorm and Residual Connections

### Model Parameters
- Number of Layers: 12
- Channel Dimensions: 768
- Vocabulary Size: 50,304
- Context Window: 2048 tokens
- Number of Attention Heads: 12
- Dropout Rate: 0.2
- Total Parameters: ~124M

### Generation Parameters
- Temperature: 0.1 - 2.0 (default: 1.3)
- Top-K: 0 - 100 (default: 50)
- Top-P: 0.0 - 1.0 (default: 0.95)
- Repetition Penalty: 1.2

## 3. System Architecture

### Local Development
```
project/
├── app.py                 # Streamlit interface
├── model.py              # Core model implementation
├── train.py              # Training script
├── requirements.txt      # Dependencies
```

### AWS Deployment Architecture
1. **Training Infrastructure**
   - SageMaker Training Jobs
   - P4d.24xlarge Instance
   - Distributed Training Support
   - S3 for Model Artifacts

2. **Inference Infrastructure**
   - SageMaker Endpoints
   - Lambda Functions
   - API Gateway (WebSocket)
   - CloudWatch Monitoring

### WebSocket Integration
- Real-time token streaming
- Bi-directional communication
- Client-side JavaScript integration
- AWS API Gateway WebSocket APIs

## 4. Training Process

### Data Preparation
- Dataset: Project Gutenberg 19 (PG19)
- Tokenization: GPT-2 tokenizer (tiktoken)
- Data Sharding: 100M tokens per shard
- Train/Validation Split

### Training Configuration
- Batch Size: 524,288 tokens
- Learning Rate: 1e-4 with cosine decay
- Warmup Steps: 100
- Weight Decay: 0.01
- Gradient Clipping: 1.0
- Mixed Precision Training: FP16

### Optimization Features
- Gradient Accumulation
- Distributed Training (DDP)
- Mixed Precision Training
- Learning Rate Scheduling
- Memory Optimization

### Training Monitoring
- Loss Tracking
- GPU Memory Monitoring
- Performance Metrics
- Training Progress Visualization

## 5. Deployment and Usage

### Local Deployment
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run Streamlit app:
```bash
streamlit run app.py
```

### AWS Deployment
1. Deploy SageMaker endpoint:
```bash
python sagemaker_serverless_endpoint_deployment.py
```

2. Configure WebSocket client:
```javascript
const socket = new WebSocket('wss://[YOUR-API-GATEWAY-URL]');
```

### Generation Parameters
The model's output can be controlled through several parameters:
- `max_tokens`: Maximum length of generated text
- `temperature`: Controls randomness (higher = more creative)
- `top_k`: Limits vocabulary choices to top K tokens
- `top_p`: Nucleus sampling threshold

## 6. Performance and Limitations

### Performance Metrics
- Training Speed: ~32K tokens/second on P4d.24xlarge
- Inference Latency: ~100ms per token
- Memory Usage: ~500MB model size

### Known Limitations
- Maximum context length of 2048 tokens
- Generation quality depends on temperature settings
- Limited to Victorian-era writing style
- Resource-intensive training requirements

## 7. Future Improvements
- Implement model quantization
- Add support for fine-tuning
- Enhance WebSocket streaming performance
- Implement caching for frequent prompts
- Add support for different literary styles

## 8. Contributing
Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## 9. License
[Specify your license here]