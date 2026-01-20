# Neurinese

**Grammarly/Copilot for Handwritten Chinese**

Chinese characters are notoriously difficult to master. Unlike English, where words can be "sounded out" phonetically, Chinese characters often lack obvious patterns between one another. Writing them relies strictly on procedural memory rather than logic.

Personally, when it comes to writing Chinese essays I often find myself forgetting, misremembering, or just straight up not knowing how to write a character. This frusttration led to a question: Can there be a "Grammarly/Copilot" that fixes the errors in a character and autocompletes your ideas seamlessly... in handwritten Chinese???

Neurinese is a real-time handwriting intelligence engine. By combining stroke-level stylistic modeling with semantic language understanding, this projects acts as a real-time "copilot" for handwritten Chinese. Here are the two main features:

1. **Smart Autocompletion** - Recognizes the context of your sentence and generates the next characters automatically in the same handwriting style

2. **Context-Aware Autocorrect** - If you write a character with a slight incorrection, it detects the error, matches it with the context of the sentence and regenerates the correct character to be used, again in the handwriting style of the user

<img src="./media/Recon.gif" width="200" />

*GIF of the model reconstructing and mimic user handwriting through teacher forcing

## Table of Contents

- [Overview](#overview)
- [Inspiration](#inspiration)
- [Model Architecture](#model-architecture)
  - [Pipeline](#pipeline)
  - [Natural Language Processing (NLP)](#natural-language-processing-nlp)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-architecture-cnn)
  - [Variational Autoencoder (VAE)](#variational-autoencoder-architecture-vae)
- [Challenges Faced](#challenges-faced)
- [Getting Started](#getting-started)
- [Milestones](#milestones)

## Overview

Neurinese combines recognition and generation to form an end-to-end handwriting intelligence pipeline:

Unlike standard OCR which sees static images, this model learns from raw stroke data:

```
(dx, dy, pen state)

pen state: 0 = pen up, 1 = pen down
```

By modeling the motion rather than the pixels, the system learns a continuous Latent Space that captures two distinct layers of information:

- **Structural information** - how characters are formed

- **Stylistic Intent** - how a person writes

This enables for an autoregressive handwriting syntehsis ability, where characters can literally be drawn by the model as if you drew it.

## Inspiration

When Apple first released their Math Notes feature back in the Summer of 2024, it especially intrigued me with how it could not only solve equations but render the solution in the user's own handwriting style.

<img src="./media/AppleMathNotes.png" width="200">

To achieve the handwriting aspect, a system must somehow undersatnd the dynamics of writing rather than simply recognizing symbols. This project aims to explore this concept in the context of handwritten Chinese, whcih inherently lacks any pattern, perfect for model memorization and handwriting synthesis.

Neurinese explores this idea in the context of Chinese handwriting by modeling characters as sequences of pen movements. This project focuses on learning stroke-level embeddings that facilitate autocompletion, synthesis, and style-aware generation.

While most character reocnigziation rely on CNN-based image recognition, the crucial question this project seeks to answer is:

> Can a model learn how characters are written, not just what they look like?

This project investigates human-centered AI and generative modeling, focusing on learning representations from the dynamics of handwriting rather than from images alone.

## Model Architecture

### Pipeline

1. User writes a sequence of characters.
2. The CNN model can partially reocgnize and match the character intended to an actual character.
3. The VAE Encoder analyzes the strokes to generate a running User Style Embedding ($z_{style}$).
4. The NLP model combines the currently drawn recognized character to analyze the sentence meaning. Two pathways emerge:
    1. If the user drew the character incorrectly given the context of the sentence. The autocorrection decision can be made.
    2. Otherwise, factoring in the current character, the model predicts the next character(s)/phrases if the occurrence is above a threshold
        - Input: `"天气非常 (The weather is very)..."`
        - Prediction: `"好 (Good)" - 85% probability`
        - Except this happens on a handwriting level
5. The system passes the Style Vector ($z_{style}$) and the predicted next character `"好 (Good)"` into the MDN Decoder.
6. The system draws the character `"好 (Good)"` using the user's specific handwriting characteristics.

### Natural Language Processing (NLP)

WIP. To understand context.

### Convolutional Neural Network Architecture (CNN)

To recognize characters.

### Variational Autoencoder Architecture (VAE)

This model learns the handwriting nuance.

The system uses a recurrent VAE architecture (similar to SketchRNN) but obvisouly modified for the high-density stroke constraints and multimodality of Chinese characters.

The pipeline consists of three core components:

1. **Bi-Directional Encoder:** A bi-directional LSTM processes the input sequence of strokes, compressing them into a fixed-length latent vector $z$ which is samples from a Gaussian distribution. This vector acts as a compressed embedding of the character.

2. **Autoregressive Decoder:** An autoregressive uni-directional LSTM conditioned on $z$ from the encoder at each step predicts the probability distribution of the next state `(dx, dy, pen state)` based on previous state and global context.

3. **Loss Functions:** The tricky part to be tackled with. Standard regression layers failed to capture handwriting complexity (see section below). Currently migrating to Mixture Density Network for stochastic sampling for a Gaussian Mixture Model.

<img src="./media/VAE_architecture.png" width="500">

## Challenges Faced

Handwriting is inherently multimodal. Especially in Chinese.

At any point in writing a character, strokes can include:

- Lifting the pen to a new place
- Smooth continuations of a stroke, and oppositely:
- Sudden sharp direction changes

Initial prototypes used standard regression layers and loss functions such as Mean Squared Error (MSE) loss collapsed these possibiilities into their mean, causing the model generation to keep converging into diagonal suqiggles. MSE loss forces the model to minimize the average error between these valid optiopns, causing the model to quite literally dodge handwriting nuance and follow through with the predicted mathematical mean of all valid paths rather than comitting to a specific stroke path.

This failure highlights a key insight that I completely missed when approaching the handwriting dynamic.

> Handwriting cannot be learned as a determininistic regression problem

Currenlty approaching an altnerative approach of stocachistic modelling to more accurately generate characters as well as preserve expressive variabiliity

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (preferably with CUDA 13.0)

### Installation

1. Install PyTorch compiled with CUDA via the [official site](https://pytorch.org/). Check NVIDIA CUDA version by running this command in terminal:

    ```bash
    nvidia-smi
    ```

2. Clone this project:

    ```bash
    git clone https://github.com/dark-sorceror/Neurinese.git

    cd neurinese

    pip install -r requirements.txt
    ```

    *Note: This includes all the necessary model weights and training data.*

3. Run the main file for prototype testing

    ```bash
    python main.py
    ```

## Milestones

- [x] Achieve some sort of model inference of the character
- [ ] Get the autoregressive inference working (learning from itself rather than being observant)
- [ ] Scale up to train on full dataset; no more training on duplicates to enforce overfitting
- [ ] Conditioning and encouraging style consistency to predict character writing in foreign characters
- [ ] Incorporate some sort of system to recognize stroke differences between the correct character and the user written character
- [ ] Integrate NLP and CNN to generate semantic understanding
- [ ]  Work toward the autocorrect inference pipeline
- [ ] Scaling and deployment
