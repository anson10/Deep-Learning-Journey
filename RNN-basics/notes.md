### Recurrent Neural Networks (RNNs)

A **Recurrent Neural Network (RNN)** is a type of neural network designed for sequential data. Unlike feedforward neural networks, RNNs have connections that form directed cycles, enabling them to "remember" previous inputs in the sequence. This makes them useful for tasks like time series prediction, natural language processing (NLP), and speech recognition, where data comes in sequences.

#### How RNNs Work

In an RNN, the output from the previous step is fed as input to the current step. The network processes input one element at a time while maintaining a hidden state that captures information from previous steps.

**Mathematical Representation**:
The hidden state at step $ t $ is updated using the formula:

$$
h_t = \sigma(W_{hx} x_t + W_{hh} h_{t-1} + b_h)
$$

Where:
- $ h_t $ = hidden state at time step $ t $
- $ W_{hx} $ = weight matrix for the current input $ x_t $
- $ W_{hh} $ = weight matrix for the previous hidden state $ h_{t-1} $
- $ b_h $ = bias term
- $ \sigma $ = activation function (often tanh or ReLU)

The output $ o_t $ at each time step $ t $ is computed as:

$$
o_t = \sigma(W_{ho} h_t + b_o)
$$

Where:
- $W_{ho} $ = weight matrix for converting the hidden state to the output
- $ b_o $ = bias term for the output layer

#### Simple ASCII Diagram of RNN

```
    x1       x2       x3       x4
     ↓        ↓        ↓        ↓
  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
  │ h1  │→ │ h2  │→ │ h3  │→ │ h4  │
  └─────┘  └─────┘  └─────┘  └─────┘
     ↓        ↓        ↓        ↓
    o1       o2       o3       o4
```

Each hidden state $ h_t $ is passed along with the next input in the sequence, and the output $ o_t $ can be produced at each step.

### Types of RNNs

1. **Many-to-One RNN**:
   - Used when we want a single output after processing the entire input sequence, e.g., sentiment analysis.

```
    x1 → x2 → x3 → x4 → h_final → output
```

2. **Many-to-Many RNN**:
   - Produces an output at each time step, e.g., in machine translation or video frame labeling.

```
    x1 → h1 → o1
    x2 → h2 → o2
    x3 → h3 → o3
    x4 → h4 → o4
```

3. **Bidirectional RNN**:
   - Has two hidden states for each time step: one processes the sequence forward, and the other processes it backward. This setup helps capture dependencies from both directions.

```
    x1 → h1_forward ← x1
         h1_backward
    x2 → h2_forward ← x2
         h2_backward
```

### Vanishing and Exploding Gradient Problems

#### Vanishing Gradient:
When training deep RNNs, gradients (the values used to update the weights during backpropagation) often become very small as they are propagated backward through time. This causes the weights to update very slowly, making it hard for the network to learn long-range dependencies.

**Explanation**:
- RNNs rely on backpropagation through time (BPTT) to adjust weights.
- Gradients are multiplied at each time step during backpropagation.
- If the values of gradients are small (e.g., between 0 and 1), repeated multiplication over many time steps causes the gradients to "vanish" (become extremely small), halting learning.

#### Exploding Gradient:
Conversely, if the gradients are large (e.g., greater than 1), multiplying them across time steps can cause them to "explode" (become extremely large). This leads to unstable updates in the network, making it difficult to converge.

**Explanation**:
- If gradients grow exponentially during backpropagation, they will become extremely large, leading to large updates in the network’s weights.
- This can cause the model to diverge during training.

### Solutions to Vanishing and Exploding Gradients

1. **Gradient Clipping**:
   - For exploding gradients, we can apply gradient clipping, which scales down gradients if they exceed a certain threshold.

2. **Long Short-Term Memory (LSTM)**:
   - LSTMs are a special type of RNN designed to combat the vanishing gradient problem. They use gating mechanisms to control the flow of information, which helps retain important information over long sequences.

3. **Gated Recurrent Units (GRU)**:
   - GRUs are a simplified version of LSTMs that also help with the vanishing gradient problem by using a similar gating mechanism but with fewer parameters.


### Summary of RNN Types

| **Type**              | **Use Case**                                        | **Characteristics**                                                                                                                                   |
|-----------------------|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| Simple RNN            | Basic sequence tasks                               | Struggles with long-term dependencies due to vanishing/exploding gradient issues.                                                                      |
| LSTM                  | Tasks requiring long-term memory                   | Uses memory cells and gates to combat vanishing gradients, ideal for long-term dependencies.                                                           |
| GRU                   | Tasks requiring less memory                        | Similar to LSTM but with fewer parameters, often performs comparably with reduced complexity.                                                          |
| Bidirectional RNN      | When future context is needed alongside past info  | Processes the input sequence in both directions, useful for tasks like speech recognition where both past and future context matter.                   |

