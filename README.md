# uav-federated-learning-optimization
Decentralized UAV Federated Learning using EfficientNet-B0, FedAvg, fog-based model filtering, and optimization algorithms (ABC, PSO &amp; GWO) for model selection and performance optimization.
# Decentralized UAV Federated Learning with Optimization

A simulation-based Federated Learning framework for UAV/drone networks that combines **EfficientNet-B0**, **FedAvg**, fog-level model filtering, and optimization-based model selection using **Artificial Bee Colony (ABC), Particle Swarm Optimization (PSO), and Grey Wolf Optimization (GWO)**.

The project explores how locally trained models from multiple UAVs can be filtered and selectively aggregated while tracking **accuracy, latency, computational cost, model acceptance, and fitness**.

---

## Project Overview

In UAV-based systems, sending every locally trained model directly to a central server can introduce communication overhead and unnecessary latency.

This project simulates a hierarchical federated learning workflow in which:

**UAVs → Routers/Fog Brokers → Cloud**

Each UAV independently trains an image-classification model on its local portion of the dataset. The resulting model updates are evaluated and passed through fog-level filtering before a subset of models is selected for aggregation.

Three optimization-based selection approaches are implemented and studied:

* **Artificial Bee Colony (ABC)**
* **Particle Swarm Optimization (PSO)**
* **Grey Wolf Optimization (GWO)**

The selected models are aggregated using **Federated Averaging (FedAvg)**.

---

## System Architecture

```text
                    ┌──────────────────┐
                    │      Cloud       │
                    │  Global Model    │
                    └────────▲─────────┘
                             │
                       FedAvg Aggregation
                             │
              ┌──────────────┴──────────────┐
              │                             │
       ┌──────┴──────┐               ┌──────┴──────┐
       │ Fog/Router A│               │ Fog/Router B│
       │  UAVs 1–5   │               │  UAVs 6–10  │
       └──────▲──────┘               └──────▲──────┘
              │                             │
        Model Filtering                Model Filtering
              │                             │
      ┌───────┴───────┐             ┌───────┴───────┐
      │ UAV │ UAV │ UAV│ ...         │ UAV │ UAV │ UAV│ ...
      └─────┴─────┴────┘             └─────┴─────┴────┘
```

Within each federated round:

1. Each UAV performs local training.
2. Local model performance and communication characteristics are recorded.
3. Fog brokers filter model updates according to model-size constraints.
4. A fitness function considers model accuracy and delay.
5. ABC, PSO, or GWO selects candidate models.
6. Selected models are aggregated using FedAvg.
7. The aggregated model is sent to the cloud/global layer.
8. The updated global model is distributed back to the UAVs.
9. Accuracy, latency, cost, acceptance rate, and fitness are logged.

---

## Dataset

The experiments use the **RSSCN7 remote-sensing image dataset**, containing **7 image classes**.

The notebooks download/load the dataset and preprocess images using:

* Resize: `224 × 224`
* Tensor conversion
* ImageNet normalization

The dataset is divided into training and testing subsets, and the training data is distributed across **10 simulated UAV clients**.

---

## Model

The classification model is based on:

**EfficientNet-B0**

The pretrained EfficientNet-B0 architecture is loaded using TorchVision weights and its final classifier is modified for the **7 RSSCN7 classes**.

All feature layers are configured for fine-tuning.

Each UAV maintains its own local copy of the model and performs local training using:

* Optimizer: Adam
* Learning rate: `0.001`
* Loss: Cross-Entropy Loss
* Batch size: `32`
* Local epochs: `5`

---

## Federated Learning

The project uses **Federated Averaging (FedAvg)** to combine selected model updates.

Rather than sharing raw training images between UAVs, the simulation works with locally trained model parameters.

The overall process is:

```text
Local Training
      ↓
Model Evaluation
      ↓
Fog-Level Filtering
      ↓
Optimization-Based Selection
      ↓
FedAvg
      ↓
Global Model
      ↓
Distribution to UAVs
```

---

## Fog-Level Model Filtering

Each fog broker evaluates incoming UAV models before aggregation.

Models are categorized into:

* **VALID** — model satisfies the size constraint
* **REPROCESS** — model falls within the intermediate size range
* **DROPPED** — model exceeds the permitted threshold

The experiments use model-size thresholds at the fog layer and divide the 10 UAVs into two groups:

* **Router/Fog A:** UAVs 1–5
* **Router/Fog B:** UAVs 6–10

This creates a hierarchical aggregation structure before the final cloud aggregation.

---

## Optimization-Based Model Selection

### 1. Artificial Bee Colony — ABC

The ABC experiment uses a fitness function based on:

* Model accuracy
* Total delay

The implementation maintains a population of candidate models and iteratively updates the candidate solutions before selecting the models with the lowest fitness.

File:

```text
fed_abc_effnetb0.ipynb
```

---

### 2. Particle Swarm Optimization — PSO

The PSO experiment evaluates candidate UAV models using the same accuracy-delay fitness formulation and selects the top-performing models for fog-level aggregation.

The implementation ranks candidate models according to their calculated fitness and selects the best candidates from each fog group.

File:

```text
fed_pso_effnetb0.ipynb
```

---

### 3. Grey Wolf Optimization — GWO

The GWO experiment uses a fitness-based model-selection mechanism to identify the best candidate UAV models according to accuracy and delay.

The selected models are then aggregated at the fog level before cloud aggregation.

File:

```text
fed_gwo_effnetb0.ipynb
```

---

## Fitness Function

The experiments use a minimization-based fitness formulation combining accuracy and communication delay:

```text
Fitness = α(1 − Accuracy) + β(Normalized Delay)
```

where:

* `α = 1.0`
* `β = 0.05`

A lower fitness value represents a preferable combination of model accuracy and delay within the implemented selection process.

---

## Simulation Configuration

| Parameter                 |           Value |
| ------------------------- | --------------: |
| Number of UAVs            |              10 |
| Federated rounds          |              10 |
| Local epochs              |               5 |
| Batch size                |              32 |
| Bandwidth                 |         50 Mbps |
| Image size                |       224 × 224 |
| Number of classes         |               7 |
| Model                     | EfficientNet-B0 |
| Aggregation               |          FedAvg |
| Optimizer                 |            Adam |
| Learning rate             |           0.001 |
| Fog groups                |               2 |
| Selected models per group |               3 |
| α                         |             1.0 |
| β                         |            0.05 |

---

## Performance Metrics

The experiments record several metrics across federated rounds:

### Global Accuracy

Accuracy of the aggregated global model on the test set.

### Total Latency

Combined simulated execution, transmission, and cloud communication delay.

### Cloud Latency

Time associated with transmitting the aggregated model to the cloud layer.

### Model Acceptance Rate

Proportion of UAV models accepted by each fog broker.

### Fitness Score

The optimization objective combining model accuracy and delay.

### Computational/Communication Cost

A simulated cost based on:

* CPU usage
* Memory/model size
* Bandwidth usage

### Local Training Loss

Training loss recorded for each UAV during each federated round.

---

## Repository Structure

```text
uav-federated-learning-optimization/
│
├── README.md
│
├── fed_abc_effnetb0.ipynb
├── fed_pso_effnetb0.ipynb
└── fed_gwo_effnetb0.ipynb
```

The three notebooks represent different optimization experiments within the same overall federated learning framework.

---

## Technologies Used

* Python
* PyTorch
* TorchVision
* NumPy
* Matplotlib
* CSV
* EfficientNet-B0
* Federated Learning
* FedAvg
* Artificial Bee Colony
* Particle Swarm Optimization
* Grey Wolf Optimization

---

## Key Concepts Demonstrated

This project brings together several areas of AI and distributed computing:

* Federated Learning
* Computer Vision
* Transfer Learning
* Deep Learning
* UAV/edge computing simulation
* Fog computing
* Model aggregation
* Optimization-based model selection
* Communication latency analysis
* Computational cost analysis
* Performance evaluation

---

## Experiments

The repository contains three optimization experiments using the same overall federated learning setup:

| Experiment | Selection Method            | Model           |
| ---------- | --------------------------- | --------------- |
| ABC        | Artificial Bee Colony       | EfficientNet-B0 |
| PSO        | Particle Swarm Optimization | EfficientNet-B0 |
| GWO        | Grey Wolf Optimization      | EfficientNet-B0 |

Each experiment produces plots for metrics such as global accuracy, latency, acceptance rate, and fitness across federated rounds.

---

## Purpose

The goal of this project is to investigate how **selective model aggregation and optimization-based filtering** can be incorporated into a UAV-oriented Federated Learning environment.

Rather than treating every local update equally, the framework evaluates model updates using accuracy, delay, model size, and cost-related factors before aggregation.

This provides a basis for studying the trade-offs between:

**Model Performance ↔ Communication Delay ↔ Resource Cost**

---

## Author

**Pranjal Jain**

B.Tech — Artificial Intelligence & Machine Learning

Govt. Mahila Engineering College, Ajmer
