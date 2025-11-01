# 🧠 Transformers Neural Network — Interactive Visualizer - Calculation Methodology

> **Live Demo:** [https://transformers.gerivan.me](https://transformers.gerivan.me)  
> **Repository:** [🧠 Transformers Interactive Visualizer](https://github.com/gerivanc/transformers)

---

## 🌐 Overview

**Transformers Neural Network** is a **fully interactive, real-time visualization** of the **Transformer architecture** running **100% in the browser** using **HTML, CSS, JavaScript, and TensorFlow.js**.

This project brings deep learning to life with:
- **Multi-Head Attention** with color-coded heads
- **Real-time training** with loss tracking
- **Forward & Backward pass animations**
- **GPT (Decoder-only) vs T5 (Encoder-Decoder) modes**
- **Live gradient flow visualization**
- **Embeddings projected in 2D**
- **Secure, copy-protected, and inspection-resistant**

---

## 🧮 Mathematical Foundation

### 🔢 Core Architecture Calculations

```javascript
// Transformer Dimensionality Calculations
const d_model = 512;        // Model dimensionality
const n_heads = 8;          // Number of attention heads
const head_dim = d_model / n_heads;  // 64 dimensions per head
const d_ff = 2048;          // Feed-forward dimension
```

**Multi-Head Attention Mechanism:**
```python
# Scaled Dot-Product Attention
def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V), attention_weights

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
```

**Positional Encoding Formula:**
```python
def positional_encoding(seq_len, d_model):
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
                        -(math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```

---

## ⚙️ Real-Time Training Algorithm

### 🔄 Forward Propagation Pipeline

**Token Embedding Process:**
```javascript
class TokenEmbedding {
    constructor(vocabSize, dModel) {
        this.embedding = tf.layers.embedding({
            inputDim: vocabSize,
            outputDim: dModel,
            inputLength: sequenceLength
        });
        this.pe = this.createPositionalEncoding(sequenceLength, dModel);
    }
    
    forward(tokens) {
        const embeddings = this.embedding.apply(tokens);
        return tf.add(embeddings, this.pe);
    }
}
```

**Training Loop Mathematics:**
```javascript
// Cross-Entropy Loss Calculation
function computeLoss(yTrue, yPred) {
    return tf.losses.softmaxCrossEntropy(
        tf.oneHot(yTrue, vocabSize), 
        yPred
    );
}

// Gradient Descent Optimization
const optimizer = tf.train.adam(0.001);
const gradients = tf.variableGrads(() => {
    const prediction = model.forward(inputSequence);
    return computeLoss(targetSequence, prediction);
});

optimizer.applyGradients(gradients.grads);
```

### 📊 Performance Metrics

**Training Convergence Analysis:**
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Optimizer**: Adam (β₁=0.9, β₂=0.999, ε=10⁻⁸)
- **Learning Rate**: 0.001 with exponential decay
- **Batch Size**: Dynamic based on sequence length
- **Gradient Clipping**: Norm-based (max_norm=1.0)

---

## 🎨 Visualization Engine Architecture

### 🔬 Multi-Head Attention Visualization

**Attention Weight Calculation:**
```javascript
class AttentionVisualizer {
    constructor(nHeads, dModel) {
        this.headColors = this.generateHeadColors(nHeads);
        this.attentionWeights = new Map();
    }
    
    // Calculate attention distribution
    calculateAttention(query, key, value, mask) {
        const d_k = query.shape[-1];
        const scores = tf.matMul(query, key.transpose([0, 1, 3, 2]))
                      .div(tf.sqrt(d_k));
        
        if (mask) {
            scores.add(mask.mul(-1e9));
        }
        
        return tf.softmax(scores, -1);
    }
    
    // Visualize attention patterns
    visualizeAttention(headWeights, sourceTokens, targetTokens) {
        const attentionMatrix = this.createAttentionMatrix(headWeights);
        this.renderAttentionHeatmap(attentionMatrix, sourceTokens, targetTokens);
        this.animateAttentionFlow(headWeights);
    }
}
```

### 🌊 Gradient Flow Animation System

**Backpropagation Visualization:**
```javascript
class GradientVisualizer {
    constructor() {
        this.gradientColors = {
            'positive': '#00ff88',  // Green for positive gradients
            'negative': '#ff6b6b',  // Red for negative gradients
            'neutral': '#00d4ff'    // Cyan for near-zero
        };
    }
    
    // Calculate gradient magnitudes
    computeGradientMagnitudes(gradients) {
        return Object.entries(gradients).map(([param, grad]) => ({
            parameter: param,
            magnitude: grad.norm().arraySync(),
            direction: grad.mean().arraySync() > 0 ? 'positive' : 'negative'
        }));
    }
    
    // Animate gradient flow
    animateGradientFlow(gradients, layers) {
        const magnitudes = this.computeGradientMagnitudes(gradients);
        this.createGradientPulses(magnitudes, layers);
        this.updateParameterColors(magnitudes);
    }
}
```

---

## 🏗️ Architecture Mode Calculations

### 🧬 GPT vs T5 Mode Switching

**GPT (Decoder-Only) Architecture:**
```javascript
class GPTArchitecture {
    constructor(config) {
        this.layers = [
            new InputEmbedding(config.vocabSize, config.dModel),
            ...Array(config.numLayers).fill().map(() => 
                new DecoderLayer(config.dModel, config.nHeads, config.dFF)
            ),
            new OutputProjection(config.dModel, config.vocabSize)
        ];
        this.mask = this.createCausalMask(config.seqLength);
    }
    
    forward(tokens) {
        let x = this.layers[0].forward(tokens);
        for (let i = 1; i < this.layers.length - 1; i++) {
            x = this.layers[i].forward(x, this.mask);
        }
        return this.layers[this.layers.length - 1].forward(x);
    }
}
```

**T5 (Encoder-Decoder) Architecture:**
```javascript
class T5Architecture {
    constructor(config) {
        this.encoder = new EncoderStack(config);
        this.decoder = new DecoderStack(config);
        this.crossAttention = new CrossAttentionLayer(config);
    }
    
    forward(encoderInput, decoderInput) {
        const encoderOutput = this.encoder.forward(encoderInput);
        const decoderOutput = this.decoder.forward(decoderInput, encoderOutput);
        return decoderOutput;
    }
}
```

### 📈 Performance Optimization Calculations

**Memory Efficiency Optimizations:**
```javascript
// Gradient Checkpointing
function checkpointedForward(model, inputs) {
    return tf.tidy(() => {
        const intermediate = [];
        let x = inputs;
        
        // Forward pass with checkpointing
        for (let i = 0; i < model.layers.length; i++) {
            if (i % checkpointInterval === 0) {
                x = tf.keep(x);
            }
            x = model.layers[i].forward(x);
            intermediate.push(x);
        }
        
        return { output: x, intermediates: intermediate };
    });
}

// Dynamic Batch Processing
class DynamicBatcher {
    constructor(maxBatchSize, maxSequenceLength) {
        this.maxBatchSize = maxBatchSize;
        this.maxSequenceLength = maxSequenceLength;
        this.currentBatch = [];
    }
    
    calculateOptimalBatch(sequences) {
        const sequenceLengths = sequences.map(seq => seq.length);
        const avgLength = sequenceLengths.reduce((a, b) => a + b) / sequences.length;
        
        // Adaptive batching based on sequence lengths
        const optimalBatchSize = Math.min(
            this.maxBatchSize,
            Math.floor(this.maxSequenceLength / avgLength)
        );
        
        return optimalBatchSize;
    }
}
```

---

## 🔒 Security & Integrity Calculations

### 🛡️ Anti-Tampering Mechanisms

**Code Integrity Verification:**
```javascript
class SecurityManager {
    constructor() {
        this.integrityHashes = new Map();
        this.initialized = this.initializeSecurity();
    }
    
    initializeSecurity() {
        // Calculate integrity hashes for critical functions
        this.criticalFunctions.forEach(func => {
            const hash = this.calculateFunctionHash(func);
            this.integrityHashes.set(func.name, hash);
        });
        
        // Set up integrity monitoring
        setInterval(() => this.verifyIntegrity(), 1000);
    }
    
    calculateFunctionHash(func) {
        const funcString = func.toString();
        return this.sha256(funcString);
    }
    
    verifyIntegrity() {
        for (const [funcName, expectedHash] of this.integrityHashes) {
            const currentHash = this.calculateFunctionHash(window[funcName]);
            if (currentHash !== expectedHash) {
                this.triggerSecurityResponse();
                break;
            }
        }
    }
}
```

### 📱 Responsive Design Calculations

**Adaptive Layout Mathematics:**
```javascript
class ResponsiveDesignEngine {
    constructor() {
        this.breakpoints = {
            mobile: 768,
            tablet: 1024,
            desktop: 1200
        };
        this.scaleFactors = this.calculateScaleFactors();
    }
    
    calculateScaleFactors() {
        const viewportWidth = window.innerWidth;
        const baseWidth = 1920; // Design reference width
        
        if (viewportWidth <= this.breakpoints.mobile) {
            return {
                tokenSize: 0.6,
                spacing: 0.7,
                fontSize: 0.8,
                animationSpeed: 1.2
            };
        } else if (viewportWidth <= this.breakpoints.tablet) {
            return {
                tokenSize: 0.8,
                spacing: 0.9,
                fontSize: 0.9,
                animationSpeed: 1.1
            };
        } else {
            return {
                tokenSize: 1.0,
                spacing: 1.0,
                fontSize: 1.0,
                animationSpeed: 1.0
            };
        }
    }
    
    adaptVisualization() {
        const factors = this.calculateScaleFactors();
        this.applyScaling('neuron', factors.tokenSize);
        this.applyScaling('connection', factors.spacing);
        this.adjustAnimationTimings(factors.animationSpeed);
    }
}
```

---

## 📊 Real-Time Analytics Engine

### 🔍 Performance Monitoring

**Training Metrics Calculation:**
```javascript
class AnalyticsEngine {
    constructor() {
        this.metrics = {
            loss: [],
            accuracy: [],
            gradientNorms: [],
            attentionDistributions: [],
            trainingSpeed: []
        };
        this.startTime = Date.now();
    }
    
    calculateTrainingMetrics(epoch, loss, accuracy, gradients) {
        const currentTime = Date.now();
        const elapsedTime = (currentTime - this.startTime) / 1000;
        
        const metrics = {
            epoch: epoch,
            loss: loss,
            accuracy: accuracy,
            gradients: {
                mean: this.calculateMeanGradient(gradients),
                std: this.calculateGradientStd(gradients),
                max: this.calculateMaxGradient(gradients)
            },
            timing: {
                epochDuration: elapsedTime,
                samplesPerSecond: this.calculateThroughput(epoch, elapsedTime)
            },
            convergence: this.calculateConvergenceRate(this.metrics.loss)
        };
        
        this.updateLiveDashboard(metrics);
        return metrics;
    }
    
    calculateConvergenceRate(lossHistory) {
        if (lossHistory.length < 2) return 0;
        
        const recentLosses = lossHistory.slice(-10);
        const convergenceRate = recentLosses.reduce((sum, loss, idx, arr) => {
            if (idx === 0) return 0;
            return sum + (arr[idx - 1] - loss) / arr[idx - 1];
        }, 0) / (recentLosses.length - 1);
        
        return convergenceRate * 100; // Return as percentage
    }
}
```

---

## ✨ Features

| 🧩 Feature | 📖 Description |
|-----------|----------------|
| **🎯 Interactive Tokens** | Click any token to see attention weights and gradients |
| **🔀 Dual Architecture Mode** | Switch between **GPT-style** (Decoder-only) and **T5-style** (Encoder-Decoder) |
| **⚙️ Real-Time Training** | Train a tiny Transformer live in your browser |
| **➡️ Animated Forward Pass** | Green pulses flow from input → output |
| **⬅️ Animated Backpropagation** | Red gradient pulses flow backward |
| **🧠 Multi-Head Attention** | 4 heads, each with unique color and behavior |
| **🧪 TensorFlow.js Powered** | Real neural network under the hood |
| **🔒 Secure by Design** | CSP, SRI, anti-inspection, anti-copy |

---

## 🚀 Live Demo

Open in your browser:  
[https://transformers.gerivan.me](https://transformers.gerivan.me)

> **No installation required** — works on desktop and mobile (WebGL 2.0+)

---

## 🕹️ How to Use

1. **Click "Treinar" (Train)** → 📉 Watch loss decrease in real time  
2. **Click "Forward"** → 🔄 See data flow through layers  
3. **Click "Backward"** → 🔁 Observe gradient propagation  
4. **Switch Mode** → 🧬 Compare GPT vs T5 architectures  
5. **Click any token** → 🔍 Inspect attention and gradient influence  

---

**© 2025 Gerivan Costa dos Santos**  
**Transformers Interactive Neural Network Visualizer © 2025**
