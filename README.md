# Jazzify: Generating music with LSTM!

## Project Overview
This project demonstrates the application of **Long Short-Term Memory (LSTM)** networks to generate jazz music solos. By training on a dataset of jazz music, the model learns patterns and structures of jazz improvisation and can generate original, creative solos. This project explores how AI can contribute to creative fields like music composition.

---

## Table of Contents
1. [Objectives](#objectives)
2. [Dataset](#dataset)
3. [Technical Details](#technical-details)
4. [Model Architecture](#model-architecture)
5. [Key Components](#key-components)
6. [Installation and Setup](#installation-and-setup)
7. [Usage](#usage)
8. [Results](#results)

---

## Objectives
- Apply **LSTM networks** to generate jazz music solos.
- Train a deep learning model on jazz music data to learn improvisation patterns.
- Explore the intersection of artificial intelligence and creativity, specifically in music composition.

---

## Dataset
The dataset used to train the model includes:
- **60 training examples**, each consisting of 30 musical values.
- **78 unique musical values** across the dataset.

### Data Preprocessing
The dataset is preprocessed into input and output sequences:
- **Input (`X`)**: Shape (60, 30, 78) - Represents 60 sequences of 30 time steps, each with 78 musical values.
- **Output (`Y`)**: Shape (30, 60, 78) - Represents the target sequences (shifted by one time step).

---

## Technical Details
- **Framework**: Keras with TensorFlow backend.
- **Model Type**: LSTM (Long Short-Term Memory) network.
- **Input Shape**: `(m, Tx, 78)`
  - `m`: Number of examples.
  - `Tx`: Sequence length (30 time steps).
  - `78`: Number of unique musical values.
- **Output Shape**: `(Ty, m, 78)`
  - `Ty`: Sequence length (30, same as Tx).

---

## Model Architecture
The model is designed to generate jazz music one value at a time, using its previous outputs as inputs. The key components are:
1. **Input Layer**: Accepts sequences of shape `(Tx, 78)`.
2. **LSTM Layer**: Contains 64 units (`n_a = 64`), learning temporal dependencies.
3. **Dense Layer**: Output layer with **softmax activation** to predict musical values.
4. **Reshape Layer**: Prepares data for the LSTM layer.
5. **Lambda Layer**: Performs transformations as required.

---

## Key Components
### Functions
- **`djmodel()`**: Builds the main LSTM model for training.
- **`music_inference_model()`**: Builds the inference model for generating music.
- **`predict_and_sample()`**: Generates new music sequences by sampling predicted values.
- **`one_hot()`**: Converts musical indices to one-hot encoded vectors.

---

## Installation and Setup

To set up and run the project locally, follow these steps:

```bash
# Clone the repository
git clone [repository_url]

# Navigate to the project directory
cd jazz-solo-improvisation

# Install dependencies
pip install -r requirements.txt
```

## Usage

Training the Model

```bash
# Import necessary modules
from your_model_file import djmodel
from keras.optimizers import Adam

# Build and compile the model
model = djmodel(Tx=30, n_a=64, n_values=78)
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X, a0, c0], list(Y), epochs=100)
```

Generating New Jazz Solos

```bash
# Import the inference model
from your_model_file import music_inference_model, predict_and_sample

# Build the inference model
inference_model = music_inference_model(Tx=30, n_a=64, n_values=78)

# Generate a new music sequence
generated_music = predict_and_sample(inference_model, X_initial, a0, c0)

```

## Results
After training the model for 100 epochs:

Loss: 12.3268
Accuracy: Varies by time step, reaching up to 100% for later time steps.
The model successfully generates jazz solos that demonstrate patterns learned from the training data.
