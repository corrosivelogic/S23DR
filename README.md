leaderboard score : {the competition is no longer accepting submissions for evaluation}


# Structured Semantic 3D Reconstruction

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Usage](#usage)
6. [Model Training](#model-training)
7. [Inference](#inference)
8. [Evaluation](#evaluation)
9. [Visualization](#visualization)

    
## Introduction

Structured Semantic 3D Reconstruction is a project aimed at transforming posed images and Structure from Motion (SfM) outputs into structured geometric representations (wireframes) from which semantically meaningful measurements can be extracted. This project leverages advanced deep learning techniques and a rich dataset of posed image features, sparse point clouds, and structured 3D wireframes.

## Project Structure

The repository contains the following files and directories:

- `EDA.ipynb`: Exploratory Data Analysis notebook to understand the dataset.
- `training.ipynb`: Notebook containing the model training pipeline.
- `script.py`: Script for running inference on the test dataset.
- `solution_utilities.py`: Utility functions for data transformation, metric calculation, and visualization.
- `README.md`: This readme file.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- Jupyter Notebook
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

### Clone the Repository

```bash
git clone https://github.com/yourusername/structured-semantic-3d-reconstruction.git
cd structured-semantic-3d-reconstruction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used for this project is a subset of the HoHo 5k dataset. It contains posed image features, sparse point clouds, and structured 3D wireframes.

### Dataset Structure

- `K`: Camera intrinsics
- `R`: Rotation matrices
- `t`: Translation vectors
- `gestalt`: Domain-specific segmentation
- `ade20k`: Standard segmentation
- `depthcm`: Depth maps
- `images`, `points3d`, `cameras`: Colmap reconstruction outputs
- `mesh_vertices`, `mesh_faces`, `face_semantics`, `edge_semantics`: Mesh data
- `wf_vertices`, `wf_edges`: Wireframe targets

## Usage

### Exploratory Data Analysis

Run the `EDA.ipynb` notebook to explore the dataset and gain insights into its structure and contents.

### Model Training

1. Load the dataset and preprocess it.
2. Define the neural network architecture.
3. Train the model using the training dataset.
4. Evaluate the model on the validation set.

Details of these steps can be found in the `training.ipynb` notebook.

### Inference

Use the `script.py` to run inference on the test dataset.

```bash
python script.py --data path/to/test_data --model path/to/saved_model
```

### Evaluation

Evaluation is performed using the Wire Frame Edit Distance (WED) metric. The `solution_utilities.py` file contains the necessary functions to calculate this metric.

### Visualization

The `solution_utilities.py` file also includes functions to visualize the 3D wireframes and point clouds. Use these functions to generate plots and analyze the reconstruction quality.

## Model Training

Open the `training.ipynb` notebook and follow these steps:

1. **Load Data:**
   ```python
   import numpy as np
   import pandas as pd

   # Load dataset
   data = pd.read_pickle('path/to/data.pkl')
   ```

2. **Define Model:**
   ```python
   from tensorflow.keras.models import Model
   from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Concatenate

   input_image = Input(shape=(256, 256, 3))
   input_depth = Input(shape=(256, 256, 1))

   x1 = Conv2D(32, (3, 3), activation='relu')(input_image)
   x1 = Flatten()(x1)

   x2 = Conv2D(32, (3, 3), activation='relu')(input_depth)
   x2 = Flatten()(x2)

   concatenated = Concatenate()([x1, x2])
   output = Dense(3, activation='linear')(concatenated)

   model = Model(inputs=[input_image, input_depth], outputs=output)
   ```

3. **Compile and Train:**
   ```python
   model.compile(optimizer='adam', loss='mean_squared_error')
   history = model.fit(train_data, epochs=10, validation_split=0.2)
   ```

4. **Evaluate:**
   ```python
   results = model.evaluate(validation_data)
   print(f'Validation Loss: {results}')
   ```

## Inference

Run the `script.py` to generate predictions for the test dataset:

```bash
python script.py --data path/to/test_data --model path/to/saved_model
```

## Evaluation

Evaluation is performed using the Wire Frame Edit Distance (WED) metric:

1. **Import Utility Functions:**
   ```python
   from solution_utilities import calculate_wed
   ```

2. **Calculate WED:**
   ```python
   wed_score = calculate_wed(predictions, ground_truth)
   print(f'WED Score: {wed_score}')
   ```

## Visualization

Use the visualization functions in `solution_utilities.py` to plot 3D wireframes and point clouds:

```python
from solution_utilities import plot_wireframe

plot_wireframe(predicted_wireframe)
```
