---
date: 2017-11-22T15:49:35-08:00
draft: false
title: "Collaborative Filtering with tensorflow"
markup: "markdown"
author: "Safak Ozkan"
---

# PREDICTION OF MOVIE RATINGS -- COLLABORATIVE FILTERING  
---

## 1. Problem Description
We are given a rating matrix $R$ where only some of the entries $R_{ij}$ are provided; otherwise rest of them are missing. The task is to predict the missing entries. As in most Machine Learning problems the assumption here is that there's an underlying stationary pattern as to how users rate the movies.

By the nature of the problem, $R$ is a sparse matrix, where the sparsity comes not from zero entries but from empty records. Therefor, we represent the training data in 3 columns: $i$: user ID , $j$: movie ID and $R_{ij}$: the rating .

| $i$ | $j$   | $R\_{ij}$ |
|:-----:|:------:|:-----:|
| 0      |   14   | 3.5   |
| 0      |  7305  | 4.0  |
| 0      |  16336 | 3.5 |
| .      |   .    |  .  |
| .      |   .    |  .  |
| 1      |   52   | 4.0 |
| 1      |   986   |  4.0  |
| 1      |   1455   |  3.5 |
| 1      |   1705   |  5.0    |
| 1      |   5598   |  4.0    |
| .      |     .   |  .    |
| .      |     .   |  .    |
| 138493 |   27278    |  5.0   |

---

## 2. Collaborative Filtering Model
The terms *Collaborative Filtering*, *Matrix Factorization* and *Low-Rank Matrix Factorization* all refer to the same recommender system model. In essence, this model is based on the assumption that users who liked the same movies are likely to feel similarly towards other movies. The term *collaborative* refers to the observation that when a large set of users are involved in rating the movies, these users are effectively collaborating to get better movie ratings for everyone because every new rating will help the algorithm learn better features for the *users-movies* system. Later, these features are used by the model to make better rating predictions for everyone else.  

The Collaborative Filtering Model can also be described as reconstructing a **low rank approximation** of matrix $R$ via its **Singular Value Decomposition** $R = U \cdot \Sigma \cdot V^T$. The low-rank reconstruction is achieved by only retaining the largest $k$ singular values, $R_k=U \cdot \Sigma_k \cdot V^T$.

**Eckart-Young Theorem** states that if $R_k$ is the best rank-$k$ approximation of $R$, then it's necessary that   
	
1. $R_k$ minimizes the Frobenius norm $||R-R_k||_F^2$, and
2. $R_k$ can be constructed by retaining only the largest $k$ singular values in the SVD formulation $R_k=U \cdot \Sigma\_k \cdot V^T$.

We can further absorb the diagonal matrix $\Sigma\_k$ into $U$ and $V$ and express the factorization as a simple dot product between the feature matrices for users and movies.
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $R\_{k(m \times n)} = U\_{(m \times k)} \cdot V\_{(k \times n)}^T$    

where, the parentheses indicate matrix size.  
$m$: number of users ($m = 138493$)  
$n$: number of movies ($n = 27278$)  
$k$: rank hyperparameter that we impose (typically $k=10$).  
$U$: user feature matrix  
$V$: movies feature matrix    

<img src="/Drawing.png" alt="Drawing" width="1000" />


Hence, we formulate the problem as an **optimization problem** and search for $U$ and $V$ by minimizing the following loss function $L$.    

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $argmin\_{\ U,V}\ L = ||R-U\cdot V^T||_F^2$.

> *An important point to make is the Frobenius norm is only a partial summation computed over the entries in $R$ where a rating is provided. In the tensorflow implementation, we don't compute the complete matrix multiplication $U \cdot V^T$ but only the dot products of $u\_i \cdot v\_j^T$ where a rating $R\_{ij}$ is provided.*

The optimization procedure searches for the values of all entries in $U$ and $V$. There are $(m+n) \times k$ many tunable variables.

The hyperparameter $k$ is to be chosen carefully by cross-validation. A small $k$ would not be enough to explain the pattern in the data adequately (*underfitting*), and too large a $k$ value would result in a model fitting on the random noise over the pattern (*overfitting*).

> *It's a worth making a brief interpretation of the feature matrices $U$ and $V$. In the $k$-rank approximation scheme, each rating $R\_{ij}$ is expressed by the dot product $u\_i \cdot v\_j^T$ ---$i^{th}$ row of $U$ and $j^{th}$ row of $V$. The goal of our optimization routine is for the model to learn a* ***latent feature representation*** *(or alternatively an* ***embedding vector***) *for each user $u\_i$, and movie $v\_j$.  By the word latent, it's implied that the features are not explicitly defined by us nor they're clearly  interpretable. Each entry in the user and movie embedding vectors $u\_i$ and $v\_j$ corresponds to an abstract feature. These features can, for instance, be the genre of the movie, or how action-filled or dramatic, entertaining or romantic the movie is or anything that would help characterize how the users rate movies.*  
> \
> *Hence, the dot product representation of the ratings $r\_{ij} = u\_i \cdot v\_j^T$ points to a linear combination of how much that feature is contained in the movie and how much that feature is favored by the user.*
> \
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $r\_{ij} = \sum\_{p=1}^k u\_{ip}v\_{jp}$

##### N.B. Explicitly  Defining Biases is not Necessary
We abstain from imposing **biases** by enforcing an extra component in $U$ and $V$ set to constant 1, since the embeddings are free to learn biases if necessary.

Since no particular bounds are imposed on the entries in the embedding vectors $u\_{i}$ and $v\_{j}$. The model is free to learn positive or negative real numbers.  

--- 
## 3. Lab41 movie ratings data
- ratings were given at intervals of 0.5: {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}

--- 

## 4. Challenges in Developing the Model on `tensorflow`
- A particular challenge in implementing a Matrix Factorization algorithm on `tensorflow` is that we can't naively pass `None` for the `shape` argument while declaring the input data tensors `R` and `R\_indices` as in `R = tf.placeholder(..., shape=(None))`. Since every rating `R[i,j]` is a function of only $2*k$ tunable variables. The `shape` parameter corresponds to how many ratings will make up a single batch in the SGD routine. To make the SGD work, I had to fix the `shape` of the `tf.placeholder` `R` and `R\_indices`  to `shape=(BATCH_SIZE, k)` instead of `shape=(None, k)`. This a small price to pay that allows me to use GPU computation and also backprop with symbolic differentiation, which gave me the flexibility to experiment in trying additional non-linear terms in the loss function without having the worry about the partial differentials with respect to the tunable variables.  

```python
R = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,))
R_indices = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE,2))
u_mean = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,1)) 
v_mean = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,1)) 
```

- At each iteration of the SGD algorithm a mini-batch of rating data $R\_{ij}$ and the corresponding user and movie index pair $(i,j)$, will be fed into the computational graph. Since each $R\_{ij}$ is represented as the dot product $u\_i \cdot v\_j^T$ by our model, this will require us to create a stack of the corresponding embedding vectors, a `U_stack` and a `V_stack` where `U_stack.getshape() == (BATCH_SIZE,k)` and `V_stack.getshape() == (BATCH_SIZE,k)`.

The implementation on `tensorflow` is a little trickier than in `numpy`, and it's implemented in the following `get_stacked_UV` module:
```python
def get_stacked_UV(R_indices, R, U, V, k, BATCH_SIZE):
    u_idx = R_indices[:,0]
    v_idx = R_indices[:,1]
    rows_U = tf.transpose(np.ones((k,1), dtype=np.int32)*u_idx)
    rows_V = tf.transpose(np.ones((k,1), dtype=np.int32)*v_idx)
    cols = np.arange(k, dtype=np.int32).reshape((1,-1))
    cols = tf.tile(cols, [BATCH_SIZE,1])

    indices_U = tf.stack([rows_U, cols], -1)
    indices_V = tf.stack([rows_V, cols], -1)

    stacked_U = tf.gather_nd(U, indices_U)
    stacked_V = tf.gather_nd(V, indices_V)
    
    return stacked_U, stacked_V
```

---

## 5. Practical Methodology
- Shuffling the data before splitting it into train, CV and test sets was crucial.

##### Splitting the Input Data:
- Training Data takes up 64% of the input data, 
- CV Data 16% and
- Test Data 20%.

## 7. Linear vs Non-linear features
- `R_pred = tf.sigmoid(R_pred) * 5` dropped the `MAE_test` approximately from `.64` to `.62`. Firstly, I can't explain why sigmoid works better--although only by a tiny bit.
- However, adding squared dot product term along with the linear dot product, didn't produce any tangible improvement. 

```python 
u_cdot_v_square = tf.square(tf.multiply(sliced_U, sliced_V)) 
nl = tf.reduce_sum(u_cdot_v_square, axis=1)
R_pred = R_pred + alpha*nl
R_pred = tf.sigmoid(R_pred) * 5
```

## 8. Linear Features:
Matrix factorization is based on a low-rank singular value decomposition (SVD).  

$$R=U \cdot V^{T}$$

An individual rating of user $i$ on movie $j$ is given by   

$$r\_{ij} = u\_{i} \cdot v\_{j}^T$$

Here, each user feature vector $u\_i$ and movie feature vector $v\_j$ is of length $k$. and the classical matrix factorization multiplies $p^{th}$ feature of $u\_{i}$ with  $p^{th}$ feature of $v\_{j}$. Here, one can assume the feature $p$ corresponds to how much of a specific genre is present in movie $j$ and how much a user $i$ likes that specific genre. When the rating $y\_{ij}$ is modeled by a dot product between $u\_i$ and $v\_j$.   

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
**=> Linear Model: MAE (CV) = 0.6237**

## 9. Nonlinear Cross-Features:
The rating prediction with cross-features 

$$r\_{ij} = u\_{i} \cdot v\_{j}^{T} + \sum\_{p=0}^{k}\sum\_{q=0}^{k} w\_{pq} (u\_{ip} \cdot v\_{jq}^{T})$$

Here, we're multiplying $p^{th}$ feature of user $i$ with $q^{th}$ feature of movie $j$. This allows the model to learn cross interactions as, say, if a user likes the actor Tom Cruise (the $p^{th}$ feature--high  value for $u\_{ip}$), and she doesn't like dark and suspenseful thrillers ($q^{th}$ feature--low value for $u\_{iq}$), however, she likes the movie Eyes Wide Shut (even though it has a high value for $v\_{jq}$), because an underlying reason that makes her not like dark suspenseful movies perhaps disappears if Tom Cruise is in the movie. For a model to capture such a pattern, it has to allow some sort of **nonlinear interactions** between feature $p$ and feature $q$.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **=> Non-linear Model: MAE (CV) = 0.6150**

```
R_pred = np.dot(U,V) + alpha1*(xft) + alpha2*(uv_sq)
```

The runtime for one epoch went from $31$ sec for linear model to $60$ sec when considering all 3 types of nonlinear feature crossings ($u\_{ip}$ ~ $v\_{jq}$), ($u\_{ip}$ ~ $u\_{iq}$), ($v\_{jp}$ ~ $v\_{jq}$)



## 10. Improvements on the Algorithm
1. The regularization term needs to take into account the average rating for each user $u\_i$, $\mu_{u_i}$. 











