# Logistic Regression Implementation from Scratch 📝🤖

An elegant, from-scratch implementation of **Logistic Regression** (both unregularized and L2-regularized) in Python using **NumPy** and **Matplotlib**. This project classifies handwritten digit images (specifically "1" vs. "5") by extracting custom intensity and symmetry features, training a logistic classifier using gradient descent, and evaluating performance with 5-fold cross-validation.

---

## 🚀 Features

* **Custom Feature Extraction**:
  * **Representation 1**: Average pixel intensity and horizontal (y-axis) symmetry.
  * **Representation 2**: Mean intensity of the middle 5 columns and vertical (x-axis) symmetry.
* **From-Scratch Logistic Regression**:
  * Custom sigmoid activation, logistic loss calculation, and gradient descent optimization.
  * Automatic convergence detection (stops when loss change is $< 10^{-5}$).
* **L2 Regularization**:
  * Includes weight decay (L2 regularization) in both loss and gradient calculations to prevent overfitting.
* **5-Fold Cross-Validation**:
  * Implements a custom 5-fold cross-validation split to tune the regularization parameter $\lambda$.
* **Rich Visualizations**:
  * Scatter plots of extracted features for both classes.
  * Loss convergence curves for various learning rates ($\alpha \in \{0.02, 0.05, 0.3, 0.8, 1.0\}$).
  * Decision boundary lines plotted directly on top of the feature space.

---

## 📂 Project Structure & Scripts

The repository contains:

### 1. `logistic_regression_implementation.ipynb`
This Jupyter Notebook contains the complete pipeline:
* **Data Loading & Reshaping**: Loads $16 \times 16$ grayscale images of digits "1" (labeled `1`) and "5" (labeled `-1`) from `./data/`.
* **Feature Engineering**:
  * *Average Intensity*: Mean pixel value across the entire image.
  * *Symmetry*: Calculated by taking the negative Euclidean norm of the difference between the image and its flipped counterpart (either horizontal or vertical).
* **Model Training & Optimization**:
  * `train_no_reg`: Trains a standard logistic regression model.
  * `train`: Trains an L2-regularized logistic regression model.
  * `test`: Evaluates accuracy on test data.
* **Hyperparameter Tuning**:
  * Compares learning rates $\alpha$ and plots the loss curves.
  * Performs 5-fold cross-validation to find the optimal regularization strength $\lambda \in \{0.05, 0.1, 0.2\}$.
* **Decision Boundary Plotting**: Computes and plots the linear decision boundary:
  $$w_1 + w_2 x_1 + w_3 x_2 = 0 \implies x_2 = -\frac{w_2}{w_3} x_1 - \frac{w_1}{w_3}$$

---

## 🧠 Mathematical Formulation

### 1. Sigmoid Function
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### 2. Logistic Loss (with L2 Regularization)
$$E(w) = \frac{1}{N} \sum_{n=1}^{N} \ln\left(1 + e^{-y_n w^T x_n}\right) + \lambda \|w\|^2$$

### 3. Gradient Update (with L2 Regularization)
$$\nabla E(w) = \frac{1}{N} \sum_{n=1}^{N} \frac{-y_n x_n}{1 + e^{y_n w^T x_n}} + 2 \lambda w$$
$$w^{(t+1)} = w^{(t)} - \alpha \nabla E(w^{(t)})$$

---

## 🛠️ Getting Started

### 1. Prerequisites
Make sure you have Python 3.8+ installed.

### 2. Installation
To run the notebook and its dependencies, install the required packages via `requirements.txt`:

```bash
pip install -r requirements.txt
```

> *Note: If you don't have a `requirements.txt` file, you can create one with the following contents:*
> ```text
> numpy
> matplotlib
> ipykernel
> ```

---

## 💻 Usage

1. Ensure your dataset files (`train_data.npy`, `train_labels.npy`, `test_data.npy`, and `test_labels.npy`) are placed in a `./data/` directory relative to the notebook.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `logistic_regression_implementation.ipynb` and run all cells to train the models, perform cross-validation, and visualize the decision boundaries.
