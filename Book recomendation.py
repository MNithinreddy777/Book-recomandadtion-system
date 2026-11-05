import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==================================================
# 1. LOAD DATASET
# ==================================================
import os

# Detect local dataset file or fall back to original path
local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Book_Recommendation_Sample_8000.csv")
file_path = local_path if os.path.exists(local_path) else r"C:\Users\mnith\OneDrive\Attachments\ML PROJECT\Book_Recommendation_Sample_8000.csv"

df = pd.read_csv(file_path)

print("=" * 65)
print("             BOOK RECOMMENDATION SYSTEM")
print("=" * 65)

required = ["User-ID", "Book-Title", "Book-Rating"]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# ==================================================
# 2. CLEAN DATA
# ==================================================

df = df.dropna(subset=required).copy()

df["User-ID"] = df["User-ID"].astype(str).str.strip()
df["Book-Title"] = df["Book-Title"].astype(str).str.strip()
df["Book-Rating"] = pd.to_numeric(
    df["Book-Rating"], errors="coerce"
)

df = df.dropna(subset=["Book-Rating"])

df = df[df["Book-Rating"].between(0, 10)]

df = df.drop_duplicates(
    subset=["User-ID", "Book-Title"],
    keep="last"
)

if df.empty:
    raise ValueError("No valid ratings found.")

print("\nDATASET INFORMATION")
print("-" * 65)
print("Total rating records:", len(df))
print("Total users:", df["User-ID"].nunique())
print("Total books:", df["Book-Title"].nunique())

# ==================================================
# 3. TRAIN-TEST SPLIT
# ==================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

print("\nTRAINING AND TESTING")
print("-" * 65)
print("Training records:", len(train_df))
print("Testing records:", len(test_df))

# ==================================================
# 4. CREATE USER-BOOK MATRIX
# ==================================================

ratings = train_df.pivot_table(
    index="User-ID",
    columns="Book-Title",
    values="Book-Rating",
    aggfunc="mean"
).fillna(0)

if len(ratings) < 2 or ratings.shape[1] < 2:
    raise ValueError(
        "The dataset must contain at least two users and two books."
    )

print("Rating matrix shape:", ratings.shape)

# ==================================================
# 5. TRAIN ALGORITHMS
# ==================================================

# Algorithm 1: KNN
knn_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

knn_model.fit(ratings)

# Algorithm 2: Cosine Similarity between users
user_vectors = ratings.to_numpy(dtype=float)

norms = np.linalg.norm(user_vectors, axis=1, keepdims=True)
normalized = np.divide(
    user_vectors,
    norms,
    out=np.zeros_like(user_vectors),
    where=norms != 0
)

cosine_similarity = normalized @ normalized.T

# Algorithm 3: Matrix Factorization using Truncated SVD
n_components = min(
    20,
    ratings.shape[0] - 1,
    ratings.shape[1] - 1
)

svd_model = None
svd_predictions = None

if n_components >= 1:
    svd_model = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )

    user_factors = svd_model.fit_transform(ratings)
    book_factors = svd_model.components_

    svd_predictions = user_factors @ book_factors

print("\nALGORITHMS TRAINED")
print("1. K-Nearest Neighbors (KNN)")
print("2. User-Based Cosine Similarity")
print("3. Truncated SVD Matrix Factorization")

# ==================================================
# 6. RATING PREDICTION FUNCTIONS
# ==================================================

def predict_knn(user_id, book_title, k=5):

    if user_id not in ratings.index:
        return np.nan

    if book_title not in ratings.columns:
        return np.nan

    user_position = ratings.index.get_loc(user_id)

    n_neighbors = min(k + 1, len(ratings))

    distances, indices = knn_model.kneighbors(
        ratings.iloc[[user_position]],
        n_neighbors=n_neighbors
    )

    neighbor_positions = indices[0][1:]
    neighbor_distances = distances[0][1:]

    neighbor_ids = ratings.index[neighbor_positions]

    book_ratings = ratings.loc[neighbor_ids, book_title]

    valid = book_ratings.to_numpy() > 0

    if not valid.any():
        return np.nan

    values = book_ratings.to_numpy()[valid]
    weights = np.maximum(1 - neighbor_distances[valid], 0)

    if weights.sum() == 0:
        prediction = values.mean()
    else:
        prediction = np.average(values, weights=weights)

    return float(np.clip(prediction, 0, 10))


def predict_cosine(user_id, book_title, k=5):

    if user_id not in ratings.index:
        return np.nan

    if book_title not in ratings.columns:
        return np.nan

    user_position = ratings.index.get_loc(user_id)

    similarities = cosine_similarity[user_position].copy()
    similarities[user_position] = -1

    neighbor_positions = np.argsort(similarities)[::-1][:k]
    neighbor_ids = ratings.index[neighbor_positions]

    values = ratings.loc[neighbor_ids, book_title]

    valid = values.to_numpy() > 0

    if not valid.any():
        return np.nan

    values = values.to_numpy()[valid]
    weights = np.maximum(
        similarities[neighbor_positions][valid], 0
    )

    if weights.sum() == 0:
        prediction = values.mean()
    else:
        prediction = np.average(values, weights=weights)

    return float(np.clip(prediction, 0, 10))


def predict_svd(user_id, book_title):

    if svd_predictions is None:
        return np.nan

    if user_id not in ratings.index:
        return np.nan

    if book_title not in ratings.columns:
        return np.nan

    user_position = ratings.index.get_loc(user_id)
    book_position = ratings.columns.get_loc(book_title)

    prediction = svd_predictions[
        user_position, book_position
    ]

    return float(np.clip(prediction, 0, 10))


algorithms = {
    "KNN": predict_knn,
    "Cosine Similarity": predict_cosine,
    "SVD": predict_svd
}

# ==================================================
# 7. EVALUATE ALL ALGORITHMS
# ==================================================

# Only test records whose user and book are present
# in the training matrix can be evaluated here.

evaluations = {}
results = []

print("\nMODEL EVALUATION")
print("-" * 65)

for name, predictor in algorithms.items():

    actual_labels = []
    predicted_labels = []

    for _, row in test_df.iterrows():

        user_id = str(row["User-ID"])
        book_title = row["Book-Title"]
        actual_rating = float(row["Book-Rating"])

        prediction = predictor(user_id, book_title)

        if pd.isna(prediction):
            continue

        # Rating >= 4 is positive; below 4 is negative.
        actual_label = int(actual_rating >= 4)
        predicted_label = int(prediction >= 4)

        actual_labels.append(actual_label)
        predicted_labels.append(predicted_label)

    if not actual_labels:
        print(f"{name}: No test records could be evaluated.")
        continue

    evaluations[name] = (
        actual_labels,
        predicted_labels
    )

    accuracy = accuracy_score(
        actual_labels, predicted_labels
    )

    precision = precision_score(
        actual_labels, predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        actual_labels, predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        actual_labels, predicted_labels,
        zero_division=0
    )

    results.append({
        "Algorithm": name,
        "Evaluated Samples": len(actual_labels),
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1-Score": round(f1, 3)
    })

    print(f"\n{name}")
    print("Evaluated samples:", len(actual_labels))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")

# ==================================================
# 8. DISPLAY CONFUSION MATRICES
# ==================================================

for name, (actual, predicted) in evaluations.items():

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    print(f"\n{name} Confusion Matrix")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    disp.plot(values_format="d")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ==================================================
# 9. COMPARE ALGORITHM PERFORMANCE
# ==================================================

if results:

    results_df = pd.DataFrame(results)

    print("\nALGORITHM COMPARISON")
    print("-" * 65)
    print(results_df.to_string(index=False))

    results_df.to_csv(
        "Book_Recommendation_Algorithm_Comparison.csv",
        index=False
    )

    results_df.set_index("Algorithm")[
        ["Accuracy (%)", "Precision", "Recall", "F1-Score"]
    ].plot(kind="bar", figsize=(10, 6))

    plt.title("Book Recommendation Algorithm Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

# ==================================================
# 10. RECOMMEND BOOKS FOR MULTIPLE USERS
# ==================================================

def recommend_books(user_id, predictor, top_n=10):

    if user_id not in ratings.index:
        print(f"User ID {user_id} is not in the training data.")
        return

    already_rated = set(
        ratings.columns[ratings.loc[user_id] > 0]
    )

    candidate_books = [
        book for book in ratings.columns
        if book not in already_rated
    ]

    predictions = []

    for book in candidate_books:

        score = predictor(user_id, book)

        if pd.notna(score):
            predictions.append((book, score))

    predictions.sort(
        key=lambda item: item[1],
        reverse=True
    )

    if not predictions:
        print("No recommendations found for this user.")
        return

    for rank, (book, score) in enumerate(
        predictions[:top_n], start=1
    ):
        print(
            f"{rank:2}. {book} "
            f"- Predicted Rating: {score:.2f}"
        )


print("\n" + "=" * 65)
print("      RECOMMEND BOOKS FOR MULTIPLE USERS")
print("=" * 65)

print("Available User IDs:", list(ratings.index[:20]))

user_input = input(
    "\nEnter multiple User IDs separated by commas: "
).strip()

selected_users = [
    user.strip() for user in user_input.split(",")
    if user.strip()
]

for user_id in selected_users:

    print("\n" + "-" * 65)
    print(f"RECOMMENDATIONS FOR USER ID: {user_id}")
    print("-" * 65)

    for algorithm_name, predictor in algorithms.items():

        print(f"\n{algorithm_name} - TOP 10 BOOKS")
        recommend_books(
            user_id,
            predictor,
            top_n=10
        )

print("\nProgram completed successfully!")
