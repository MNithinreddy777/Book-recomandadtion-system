# 📚 Book Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c.svg?logo=python&logoColor=white)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning **Book Recommendation System** implementing and benchmarking three collaborative filtering and matrix factorization algorithms: **K-Nearest Neighbors (KNN)**, **User-Based Cosine Similarity**, and **Truncated Singular Value Decomposition (SVD)**. 

The project delivers personalized Top-N book recommendations for single or multiple users simultaneously, complete with rigorous classification evaluation metrics (Accuracy, Precision, Recall, F1-Score), confusion matrices, and comparative visual analytics.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Dataset Details](#-dataset-details)
- [Algorithms Explained](#-algorithms-explained)
- [Model Evaluation & Metrics](#-model-evaluation--metrics)
- [Visualizations](#-visualizations)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Sample Execution & Output](#-sample-execution--output)
- [Contributing & License](#-contributing--license)

---

## 🌟 Overview

Recommendation engines are vital components in modern digital libraries and e-commerce platforms. This project explores user rating behaviors to predict how strongly an individual will rate unseen books and surface the most relevant reading choices.

Key highlights of this implementation:
- **Multi-Algorithm Comparison:** Compares memory-based collaborative filtering against model-based matrix factorization.
- **Data Preprocessing & Cleaning:** Robust validation, type normalization, duplicate pruning, and sparse matrix generation.
- **Evaluation Framework:** Transforms rating predictions into binary recommendation decisions ($\ge 4$ stars as positive preference) and computes confusion matrices and F1-scores.
- **Interactive Multi-User CLI:** Enables querying customized Top-10 book suggestions across multiple user IDs simultaneously.

---

## 🚀 Key Features

- **Data Pipeline:**
  - Automated cleaning, whitespace trimming, and duplicate handling.
  - Rating boundary enforcement ($0 \le \text{Rating} \le 10$).
  - Dynamic user-item interaction pivot matrix with fallback validation.
- **Three Core Machine Learning Models:**
  1. **KNN (Cosine Metric, Brute-force):** Distance-weighted neighbor aggregation.
  2. **User-Based Cosine Similarity:** Vectorized L2-normalized dot-product similarity matrix.
  3. **Truncated SVD (Matrix Factorization):** Latent factor decomposition reconstructing a dense rating prediction matrix.
- **Comprehensive Evaluation Suite:**
  - Accuracy, Precision, Recall, and F1-Score benchmarking across all algorithms.
  - Matplotlib `ConfusionMatrixDisplay` visual heatmaps for each model.
  - Grouped bar chart comparing metric trade-offs across algorithms.
  - Automated export of benchmark results to `Book_Recommendation_Algorithm_Comparison.csv`.
- **Top-N Recommendation Engine:**
  - Recommends the highest predicted rating books excluding titles the user has already read.
  - Multi-user batch lookup via a user-friendly console prompt.

---

## 🏗️ Project Architecture

```
 Raw Dataset (.csv)
        │
        ▼
 Data Cleaning & Validation (Range [0, 10], Deduplication, Null check)
        │
        ▼
 Train / Test Split (70% Training / 30% Testing)
        │
        ▼
 User-Book Interaction Matrix (Pivot Table: Users x Books)
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
1. K-Nearest Neighbors   2. Cosine Similarity   3. Truncated SVD
   (Cosine, Brute-force)    (Normalized Dot)       (Latent Factors)
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                               ▼
 Model Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
                               │
                               ▼
 Comparative Analytics & Visualizations (Bar Charts & Heatmaps)
                               │
                               ▼
 Interactive Multi-User Top-10 Book Recommendation Engine
```

---

## 📊 Dataset Details

The repository includes two datasets designed for collaborative filtering and extended metadata analysis:

### 1. `Book_Recommendation_Sample_8000.csv` (Primary Training Dataset)
A clean dataset of 8,000+ interaction records used directly by the training script:
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `User-ID` | Integer / String | Unique identifier of the reader |
| `Book-Title` | String | Title of the rated book |
| `Book-Rating` | Float / Integer | Explicit rating scale from 0 to 10 |

### 2. `Book_Recommendation_10000_With_Customer_Reviews.csv` (Enriched Metadata Dataset)
A dataset of 10,000+ interactions containing rich contextual features:
| Column Name | Description |
| :--- | :--- |
| `Interaction-ID` | Unique interaction identifier |
| `User-ID` | Unique reader identifier |
| `ISBN` | Standard book identification code |
| `Book-Title` | Book title |
| `Book-Author` | Author name |
| `Genre` | Fiction, Fantasy, Romance, Sci-Fi, Mystery, Thriller, etc. |
| `Year-Of-Publication` | Publication year |
| `Publisher` | Publishing company |
| `Book-Rating` | Numerical rating (0 to 10) |
| `Review-Summary` | Short sentiment / review heading |
| `Review-Text` | Detailed user critique / review feedback |

---

## 🧠 Algorithms Explained

### 1. K-Nearest Neighbors (KNN)
- **Concept:** Identifies the $K$ users with interaction profiles closest to the active user using cosine distance.
- **Distance Metric:** $\text{Cosine Distance} = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$
- **Rating Estimation:** Computes a weighted average of neighbor ratings using distance-based inverse weights:
  $$w_i = \max(1 - d_i, 0)$$
  $$\hat{r}_{u, b} = \frac{\sum_{i \in \text{Neighbors}} w_i \cdot r_{i, b}}{\sum_{i \in \text{Neighbors}} w_i}$$

### 2. User-Based Cosine Similarity
- **Concept:** Computes an all-pairs similarity matrix across user vectors using normalized dot products:
  $$\hat{\mathbf{U}} = \frac{\mathbf{U}}{\|\mathbf{U}\|_2}, \quad \mathbf{S} = \hat{\mathbf{U}} \hat{\mathbf{U}}^T$$
- **Rating Estimation:** Selects the top $K$ users with highest similarity scores to estimate target book ratings:
  $$\hat{r}_{u, b} = \frac{\sum_{i \in \text{Top-}K} \max(S_{u, i}, 0) \cdot r_{i, b}}{\sum_{i \in \text{Top-}K} \max(S_{u, i}, 0)}$$

### 3. Truncated SVD (Matrix Factorization)
- **Concept:** Decomposes the high-dimensional sparse user-book matrix into lower-dimensional latent representations:
  $$\mathbf{R} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$
  - $\mathbf{U}_k$: User latent feature matrix ($m \times k$)
  - $\mathbf{\Sigma}_k \mathbf{V}_k^T$: Book latent feature matrix ($k \times n$)
- **Prediction:** The full reconstructed rating matrix is calculated via dense matrix multiplication:
  $$\hat{\mathbf{R}} = \text{clip}\left(\mathbf{U}_k \mathbf{V}_k^T, 0, 10\right)$$
  This enables rapid $O(1)$ rating lookup for any user-book pair.

---

## 📈 Model Evaluation & Metrics

To evaluate recommendation performance on held-out test data, predicted continuous ratings are converted to binary recommendation labels:
- **Positive Recommendation (1):** $\text{Rating} \ge 4.0$ (User liked/appreciated the book)
- **Negative Recommendation (0):** $\text{Rating} < 4.0$ (User disliked or rated unfavorably)

### Metrics Calculated:
- **Accuracy:** Overall percentage of correct classification decisions:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **Precision:** Ability of the system to recommend only truly relevant books:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall:** Ability of the system to discover all relevant books the user liked:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **F1-Score:** Harmonic balance between Precision and Recall:
  $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 📊 Visualizations

The script automatically generates:
1. **Confusion Matrix Heatmaps:** Generated for each algorithm via Matplotlib `ConfusionMatrixDisplay`, plotting True Positives, False Positives, True Negatives, and False Negatives.
2. **Algorithm Comparison Bar Chart:** Compares Accuracy (%), Precision, Recall, and F1-Score side-by-side.
3. **CSV Export:** Saves a consolidated benchmarking file named `Book_Recommendation_Algorithm_Comparison.csv`.

---

## 📁 Repository Structure

```
Book-recomandadtion-system/
│
├── Book recomendation.py                              # Main executable Python script
├── Book_Recommendation_Sample_8000.csv                # Primary sample dataset (8,000+ ratings)
├── Book_Recommendation_10000_With_Customer_Reviews.csv # Extended dataset with customer reviews
├── requirements.txt                                   # Python dependencies list
├── .gitignore                                         # Git ignore configuration
└── README.md                                          # Detailed project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/MNithinreddy777/Book-recomandadtion-system.git
cd Book-recomandadtion-system
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run

Execute the main recommendation script:
```bash
python "Book recomendation.py"
```

### Execution Flow:
1. The script loads the dataset and prints summary statistics (total records, unique users, unique books).
2. Performs a 70/30 train-test split and constructs the interaction matrix.
3. Trains KNN, Cosine Similarity, and Truncated SVD models.
4. Evaluates all models against the test dataset and outputs metrics to the console.
5. Displays confusion matrix plots and comparison bar charts (close chart windows to continue execution).
6. Prompts you to enter one or more user IDs to generate personalized Top-10 recommendations.

---

## 🖥️ Sample Execution & Output

```text
=================================================================
             BOOK RECOMMENDATION SYSTEM
=================================================================

DATASET INFORMATION
-----------------------------------------------------------------
Total rating records: 8000
Total users: 494
Total books: 250

TRAINING AND TESTING
-----------------------------------------------------------------
Training records: 5600
Testing records: 2400
Rating matrix shape: (494, 250)

ALGORITHMS TRAINED
1. K-Nearest Neighbors (KNN)
2. User-Based Cosine Similarity
3. Truncated SVD Matrix Factorization

MODEL EVALUATION
-----------------------------------------------------------------

KNN
Evaluated samples: 735
Accuracy: 88.44%
Precision: 0.912
Recall: 0.957
F1-Score: 0.934

Cosine Similarity
Evaluated samples: 735
Accuracy: 88.71%
Precision: 0.915
Recall: 0.959
F1-Score: 0.936

SVD
Evaluated samples: 735
Accuracy: 89.12%
Precision: 0.919
Recall: 0.961
F1-Score: 0.940

=================================================================
      RECOMMEND BOOKS FOR MULTIPLE USERS
=================================================================
Available User IDs: ['1000', '1001', '1002', '1003', '1004', ...]

Enter multiple User IDs separated by commas: 1000, 1001

-----------------------------------------------------------------
RECOMMENDATIONS FOR USER ID: 1000
-----------------------------------------------------------------

KNN - TOP 10 BOOKS
 1. Sample Book 0048 - Predicted Rating: 8.75
 2. Sample Book 0192 - Predicted Rating: 8.50
 3. Sample Book 0064 - Predicted Rating: 8.42
...

SVD - TOP 10 BOOKS
 1. Sample Book 0048 - Predicted Rating: 8.91
 2. Sample Book 0105 - Predicted Rating: 8.63
 3. Sample Book 0192 - Predicted Rating: 8.55
...

Program completed successfully!
```

---

## 🤝 Contributing & License

Contributions, feedback, and enhancements are always welcome! Feel free to fork the repository, open an issue, or submit a pull request.

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
