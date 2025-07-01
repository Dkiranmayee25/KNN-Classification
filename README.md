# KNN-Classification
# K-Nearest Neighbors (KNN) Classification - Iris Dataset 
This project demonstrates the use of the **K-Nearest Neighbors (KNN) algorithm for classification using the classic Iris dataset.

Tools and Libraries Used
- Python
- Pandas, NumPy
- Scikit-learn (KNN, preprocessing, metrics)
- Matplotlib & Seaborn (for visualization)

Dataset:
The dataset consists of 150 records of iris flowers with 4 features:
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm

Target variable: `Species` (Setosa, Versicolor, Virginica)

Project Steps
1. Data Preprocessing:
   - Dropped `Id` column (if present)
   - Label encoded the target class (`Species`)
   - Standardized feature values

2. Model Training:
   - Split into train and test sets (80/20)
   - Trained KNN for `K=1` to `10`
   - Chose the best `K` based on highest accuracy

3. Evaluation:
   - Accuracy vs K plot
   - Confusion matrix (heatmap)
   - Classification report

4. Visualization:
   - 2D decision boundaries using PetalLengthCm and PetalWidthCm

Results
- Accuracy: Up to 100% for best `K`
- Clear class separation for petal-based decision boundary

Files:
- `knn_iris_classifier.py`: Complete Python script
- `KNN_Iris_Classification.ipynb`: Jupyter Notebook version
- `Iris.csv`: Dataset (not included — download from [Kaggle](https://www.kaggle.com/datasets/uciml/iris))

Concepts Covered
- Supervised Learning
- KNN Classifier
- Feature Scaling
- Model Evaluation Metrics
- Decision Boundary Plotting

How to Run
1. Upload `Iris.csv` to the same directory or `/content` if using Google Colab.
2. Run either the `.py` script or `.ipynb` notebook.
3. View plots and evaluation results
