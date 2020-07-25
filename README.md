# Movie Recommendation Analysis
This repository analyzes the movie recommendation dataset by Probabilistic Matrix Factorization.

# datasets
MovieLens Latest Datasets are downloaded here https://grouplens.org/datasets/movielens/
In the analysis, small dataset (100k ratings applied to 9,000 movies by 600 users) is tested. To test model capability quickly, partial movies with id less than 400 (super-mini-MovieLens) are tested.

# model
Probabilistic Matrix Factorization is trained by MAP (maximum a posterior) estimation.
The goal of Matrix Factorization is to find two matrices $U$ and $V$, approximating targeted matrix R by $R = UV$

Based on the Columbia University Prof. John Paiskley's lecture (http://www.columbia.edu/~jwp2128/Teaching/W4721/Spring2017/slides/lecture_3-30-17.pdf) and their page 19, MAP inference is estimated by coordinate ascent algorithm.

<img src="https://latex.codecogs.com/svg.latex?u_i%20=%20\left(\lambda%20\sigma^2%20I%20+%20\sum_{j%20\in%20\Omega_{u_i}}%20v_j%20v_j^T%20\right)^{-1}%20\left(%20\sum_{j%20\in%20\Omega_{u_i}}%20M_{ij}%20v_j\right)" alt="q_{U}"/>
<img src="https://latex.codecogs.com/svg.latex?v_i%20=%20\left(\lambda%20\sigma^2%20I%20+%20\sum_{i%20\in%20\Omega_{v_j}}%20u_j%20u_j^T%20\right)^{-1}%20\left(\sum_{i%20\in%20\Omega_{v_j}}%20M_{ij}%20u_j%20\right)" alt="q_{V}"/>

Under PMF, model is coded under the class inheritance of sklearn BaseEstimator.
To deal with sparse matrix, math operation relies on scipy.sparse lil_matrix and coo_matrix

# Notebooks
## hyperparameter sensitivity
Leveraging GridSearchCV, two hyperparameters (features dimension, reguralization term of variance of matrix R =$\lambda \sigma^2$) are tuned by 3 folds cross valication.

Grid search is computed with 4 processes in 16GB CPU mac book pro and it takes 17min 12s to test 24 combinations in total.

# Submission
This is submitted to Statistical Machine Learning coarse (2020 Summer) in the University of Tokyo