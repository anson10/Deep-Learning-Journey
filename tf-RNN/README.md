# Recurrent Neural Networks (RNN)

## Definition and Core Concept
A Recurrent Neural Network (RNN) is a type of neural network designed to process sequential data by maintaining an internal memory state. Unlike traditional neural networks, RNNs can:
- Handle variable-length input sequences
- Retain information about previous inputs
- Share parameters across different time steps

### Core Architecture
- Input Layer: Processes sequential data (e.g., time series, text)
- Recurrent Hidden Layer: Maintains state information
- Output Layer: Generates predictions based on current state

## Mathematical Representation

### Basic RNN Equation
h(t) = tanh(W_hh * h(t-1) + W_xh * x(t) + b_h)

Where:
- h(t): Hidden state at time t
- x(t): Input at time t
- W_hh: Hidden-to-hidden weight matrix
- W_xh: Input-to-hidden weight matrix
- b_h: Hidden bias term
- tanh: Non-linear activation function

## Key Characteristics

### Strengths
- Handles sequential data naturally
- Shares weights across time steps
- Computationally efficient

### Limitations
- Vanishing Gradient Problem
- Difficulty capturing long-term dependencies
- Limited memory retention

## Long Short-Term Memory (LSTM)

### Core Innovation
LSTMs address RNN limitations by introducing gating mechanisms:

- Input Gate: Controls new information
- Forget Gate: Removes irrelevant information
- Output Gate: Determines hidden state output

### Mathematical Model
C(t) = f(t) * C(t-1) + i(t) * tanh(W_c * [h(t-1), x(t)] + b_c)

## Gated Recurrent Unit (GRU)

### Simplified Architecture
- Update Gate: Determines information retention
- Reset Gate: Controls how much past information to forget
- Computationally more efficient than LSTM

## Comparative Analysis

| Network | Complexity | Memory Handling | Computational Cost |
|---------|------------|-----------------|-------------------|
| RNN | Low | Poor | Lowest |
| LSTM | High | Excellent | Highest |
| GRU | Medium | Good | Moderate |