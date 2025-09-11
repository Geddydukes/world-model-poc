# World Model Proof of Concept

A multimodal AI system that learns to understand and remember daily experiences by processing visual and audio data. This system builds episodic memories from your daily life through self-supervised learning, enabling semantic search across your personal experiences.

## Overview

This project implements a personal AI assistant that continuously learns from your daily activities by watching and listening to your environment. It uses state-of-the-art self-supervised learning techniques to build a searchable memory system without requiring any manual labeling.

## Key Features

- **Multimodal Learning**: Processes both visual and audio data simultaneously
- **Self-Supervised Training**: Learns representations without human labels
- **Episodic Memory**: Builds searchable memories of daily experiences
- **Cross-Modal Retrieval**: Find related content across different modalities
- **Daily Processing**: Automated nightly training on new data
- **Semantic Search**: Query your memories using natural language concepts

## Architecture

### Vision Module (JEPA)
- **Joint Embedding Predictive Architecture** for visual understanding
- Student-teacher network with exponential moving average updates
- Masked token prediction for self-supervised learning
- Vision Transformer encoder with patch embedding

### Audio Module (CPC)
- **Contrastive Predictive Coding** for audio understanding
- Log-mel spectrogram preprocessing at 16kHz
- Convolutional encoder + GRU for temporal modeling
- Future frame prediction for temporal understanding

### Multimodal Alignment
- CLIP-style contrastive learning between modalities
- Projects audio and visual embeddings to shared space
- Learns cross-modal correspondences automatically

### Memory System
- SQLite database for metadata and indexing
- NumPy arrays for efficient embedding storage
- Cosine similarity search across all memories
- Date-based organization and retrieval

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd world-model-poc
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (for video processing):
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   ```

## Usage

### 1. Data Ingestion

Process raw video files into training data:

```bash
python scripts/ingest_day.py --date 2025-01-15 --clip_seconds 3 --frame_rate 1
```

This will:
- Process videos from `data/raw/2025-01-15/`
- Split into 3-second clips
- Extract frames at 1 FPS
- Extract audio segments at 16kHz
- Save to organized directories

### 2. Training

Run the nightly training pipeline:

```bash
python sleep.py --config configs/default.yaml --date 2025-01-15
```

This performs:
- Vision model training (JEPA)
- Audio model training (CPC)
- Multimodal alignment training
- Memory storage of learned embeddings

### 3. Memory Query

Search your memories for similar content:

```bash
python scripts/query_memory.py --query_path path/to/image.jpg --topk 5
```

## Configuration

Edit `configs/default.yaml` to customize:

- **Data paths**: Where to find input videos and save outputs
- **Model architecture**: Embedding dimensions, transformer depth, etc.
- **Training parameters**: Learning rates, batch sizes, training steps
- **Memory settings**: Database paths, clustering parameters

## Data Structure

```
data/
├── raw/{date}/           # Input video files (.mp4)
├── clips/{date}/         # 3-second video clips
├── frames/{date}/        # Extracted frames (.jpg)
└── audio/{date}/segments/ # Audio segments (.wav)

memory/
├── episodic.sqlite       # Metadata database
└── embeddings/           # Learned embeddings (.npy)

checkpoints/daily/        # Model checkpoints
reports/                  # Training reports
```

## Technical Details

### Vision Processing
- Input: RGB images (224×224)
- Architecture: Vision Transformer with 6 layers, 4 heads
- Embedding dimension: 256
- Masking ratio: 60% for self-supervised learning

### Audio Processing
- Input: 16kHz mono audio
- Features: 64-bin log-mel spectrograms
- Architecture: CNN encoder + GRU context network
- Hidden dimension: 256

### Training Schedule
- Vision: 5,000 steps
- Audio: 4,000 steps  
- Alignment: 2,000 steps
- Batch size: 16 (with micro-batching)
- Learning rate: 1e-4

## Memory Operations

The episodic memory system provides:

- **Storage**: Automatic embedding storage during training
- **Retrieval**: Similarity search across all stored memories
- **Organization**: Date-based and modality-based filtering
- **Persistence**: SQLite + NumPy for efficient storage

## Example Workflow

1. **Morning**: Record your day with video/audio
2. **Evening**: Run data ingestion to process raw footage
3. **Night**: Execute training pipeline to learn from new data
4. **Next day**: Query memories to find similar past experiences

## Requirements

- Python 3.8+
- PyTorch 2.2+
- FFmpeg for video processing
- 8GB+ RAM recommended
- GPU support (MPS/CUDA) for faster training

## Research Background

This implementation combines several cutting-edge self-supervised learning techniques:

- **JEPA**: Joint Embedding Predictive Architecture for visual representation learning
- **CPC**: Contrastive Predictive Coding for audio representation learning  
- **CLIP**: Contrastive Language-Image Pre-training for multimodal alignment
- **Episodic Memory**: Inspired by cognitive science research on human memory

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{world-model-poc,
  title={World Model Proof of Concept: Multimodal Episodic Memory},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```
