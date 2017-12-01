---
date: 2017-11-18T15:49:35-08:00
draft: false
title: "A Collaborative Filtering Model on tensorflow with Nonlinear Cross Features"
markup: "markdown"
author: "Safak Ozkan"
---

---

## 1. Problem Description
We are given a rating matrix $R$ where only a small fraction of the entries $R_{ij}$ are provided; otherwise the rest is missing. The task is to predict those missing entries. As in most Machine Learning problems the assumption here is that there's an underlying stationary pattern as to how users rate the movies.

By the nature of the problem, $R$ is a sparse matrix, where the sparsity comes not from zero entries but from empty records. Therefor, we represent the training data in 3 columns: $i$: user ID , $j$: movie ID and $R_{ij}$: the rating (see Table 1).    
   

<font size="+1"><strong><p align="center">Table 1. Ratings data ml-20m sparse format</p></strong></font>


| $i$: user ID | $j$: movie ID   | $R_{ij}$: the rating |
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
The terms *Collaborative Filtering*, *Matrix Factorization* and *Low-Rank Matrix Factorization* all refer to the same recommender system model. In essence, this model is based on the assumption that users who liked the same movies are likely to feel similarly towards other movies. The term *collaborative* refers to the observation that when a large set of users are involved in rating the movies, these users are effectively collaborating to get better movie ratings for everyone because every new rating will help the algorithm learn better features for the *users-movies* system. Later, these features are used by the model to make better rating predictions for everyone.  

The Collaborative Filtering Model can also be described as reconstructing a **low rank approximation** of matrix $R$ via its **Singular Value Decomposition** $R = U \cdot \Sigma \cdot V^T$. The low-rank reconstruction is achieved by only retaining the largest $k$ singular values, $R_k=U \cdot \Sigma_k \cdot V^T$.

**Eckart-Young Theorem** states that if $R_k$ is the best rank-$k$ approximation of $R$, then it's necessary that:  
 
&emsp;&emsp;&emsp;   1. $R_k$ minimizes the Frobenius norm $||R-R_k||_F^2$ and   
&emsp;&emsp;&emsp;   2. $R_k$ can be constructed by retaining only the largest $k$ singular values in $\Sigma\_k$ of the SVD formulation.

We can further absorb the diagonal matrix $\Sigma\_k$ into $U$ and $V$ and express the factorization as a simple dot product between the feature matrices for users and movies.
 
<p align="center">$R_{k(m \times n)} = U_{(m \times k)} \cdot V_{(k \times n)}^T$</p>

where, the parentheses indicate matrix size.  
$m$: number of users ($m = 138493$)  
$n$: number of movies ($n = 27278$)  
$k$: rank hyperparameter that we impose (typically $k=10$).  
$U$: user feature matrix  
$V$: movies feature matrix    

<img src="/Drawing.png" alt="Drawing" width="1000" />
<font size="+1"><strong><p align="center">Figure 1. A conceptual sketch of the Ratings data matrix $R$ decomposed into its factors: user feature matrix, $U$, and movie feature matrix, $V$. Dots in the figure, $\cdot$, illustrate given values; and question marks, $?$, the missing values. Each entry $R_{ij}$ is expressed as a dot product of the user and movie embedding vectors $U_i$ and $V_j$, respectively.</strong></font>  

Hence, we formulate the problem as an **optimization problem** and search for $U$ and $V$ by minimizing the following loss function $L$.    

$$argmin_{\ U,V}\ L = ||R - U \cdot V^T||_F^2$$

> *It's important to note that the Frobenius norm is computed only as a partial summation over the entries in $R$ where a rating is provided.*

The optimization procedure searches for the values of all entries in $U$ and $V$. There are $(m+n) \times k$ many tunable variables.

The hyperparameter $k$ is to be chosen carefully by cross-validation. A small $k$ would not be enough to explain the pattern in the data adequately (*underfitting*), and too large a $k$ value would result in a model fitting on the random noise over the pattern (*overfitting*).

> *It's worth making a brief interpretation of the feature matrices $U$ and $V$. In the $k$-rank approximation scheme, each rating $R\_{ij}$ is expressed by the dot product $U\_i \cdot V\_j^T$ (see Figure 1). The goal of our optimization routine is for the model to learn a* ***latent feature representation*** *(or alternatively an* ***embedding vector***) *for each user and movie.*   
> \
> *The term* ***latent*** *implies that the features are not explicitly defined by us nor they're definitively interpretable once the embeddings are learned by the model. Each entry in the embedding vectors $U\_i$ and $V\_j$ corresponds to the weight coefficient of an abstract feature. These features can specify the movie genre,  how action-filled or dramatic or any other distinguishing quality that would help characterize how the users rate movies. Hence, the dot product representation of the ratings $R\_{ij} = U\_i \cdot V\_j^T$ points to a* ***linear combination*** *of   
> \
> &emsp;&emsp;&emsp; 1. how much that feature is contained in the movie, and    
> &emsp;&emsp;&emsp; 2. how much that feature is favored by the user.*   

## 3. Practical Methodology  

##### MovieLens 20M dataset
- [MovieLens dataset](https://grouplens.org/datasets/movielens/20m/) data set consists of 20,000,263 ratings from 138,493 users on 27,278 movies.
- All ratings are given at intervals of 0.5:  
{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}
- Shuffling the data before splitting it into training, CV and test sets was crucial.
- Splitting the Input Data:
	- Training Data takes up 64% of the input data, 
	- CV Data 16% and
	- Test Data 20%.
- We abstain from imposing a **bias term** by enforcing an extra component set to constant $1$ in $U$ and $V$, since the embeddings are free to learn biases if necessary.
- Since no particular bounds are imposed on the entries in the embedding vectors $U\_{i}$ and $V\_{j}$. The model is free to learn positive or negative real numbers.  

<div style="width:700 px">
	<div style="float:left; width:360">
		<img src="/fig1.png" alt="fig1" width="360" />
	</div>
	<div style="float:left; width:300">
		<img src="/fig2.png" alt="fig2" width="360" />
	</div>
	<div style="float:left; width:300">
		<img src="/fig3.png" alt="fig3" width="360" />
	</div>
	<div style="float:left; width:300">
		<img src="/fig4.png" alt="fig4" width="360" />
	</div>
	<div style="clear:both"></div>
</div>
<br>
<font size="+1"><b><p align="center">Figure 2. Histogram of (*a*) all ratings in ml-20m data (*b*) mean of ratings per user \(*c*) mean of ratings per movie, and (*d*) 
	number of ratings provided by users. Minimum number of ratings provided by a user is 20, and maximum is 9254 ratings.
</p></b></font>


---   

## 4. Challenges in Developing the Model on `tensorflow`
- A particular challenge in implementing a Matrix Factorization algorithm on `tensorflow` is that we can't naively pass `None` for the `shape` argument while declaring the input data tensors `R` and `R_indices` as in `R = tf.placeholder(..., shape=(None))`.  The `shape` parameter corresponds to the number of ratings a single batch in the SGD step contains. To make the SGD work, I had to fix the `shape` of the `tf.placeholder` variables `R` and `R_indices`  to `shape=(BATCH_SIZE, k)` instead of `shape=(None, k)`.  This is a small price to pay that allows me to use `tensorflow` which provides me GPU computation and also backprop with symbolic differentiation. This gave me the flexibility to experiment with additional nonlinear terms in loss function without having the worry about the partial differentials with respect to the tunable variables. 
```python
R = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,))
R_indices = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE,2))
u_mean = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,1)) 
v_mean = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,1)) 
```

- At each SGD step a mini-batch of rating data $R_{ij}$ and the corresponding user-movie index pairs $(i,j)$ are fed into the computational graph. Since each $R\_{ij}$ is represented as the dot product $U_i \cdot V_j^T$, we have to stack the corresponding embedding vectors into 2-D tensors `U_stack` and `V_stack` where both `U_stack.getshape()` and `V_stack.getshape()` equal to `(BATCH_SIZE,k)`.   
The implementation of stacking tensors  on`tensorflow` is a little trickier than in `numpy`. It's done like this:  
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

## 5. Linear and Nonlinear Features
- `R_pred = tf.sigmoid(R_pred) * 5` dropped the `MAE_test` approximately from `.64` to `.62`. The reason for it is that the error rate, in the absence of sigmoid activation, some predictions can get bigger than 5 and some smaller than 0. Sigmoid activation brings the predictions closer to their actual values. 

- However, adding squared dot product term along with the linear dot product, didn't produce any tangible improvement and was discarded from the final model.   
```python 
u_cdot_v_square = tf.square(tf.multiply(U_stack_, V_stack)) 
nl = tf.reduce_sum(u_cdot_v_square, axis=1)
R_pred = R_pred + alpha*nl
R_pred = tf.sigmoid(R_pred) * 5
```

## 6. Cross Features:
- In the linear collaborative filtering model, $$R\_{ij} = U\_{i} \cdot V\_{j}^T$$ $p^{th}$ feature of $U\_{i}$ is multiplied with the corresponding $p^{th}$ feature of $V\_{j}$.  
$$=> Linear\ Model: MAE (CV) = 0.6237$$

- The rating prediction with cross-features:
$$R\_{ij} = U\_{i} \cdot V\_{j}^{T} + \sum\_{p=0}^{k}\sum\_{q=0}^{k} w\_{pq} (U\_{ip} \cdot V\_{jq}^{T})$$

- Here, feature $p$ of user $i$ is getting multiplied by feature $q$ of movie $j$. This allows the model to learn cross interactions as, for instance, if a user likes the actor Tom Cruise (that corresponds a high value for $U\_{ip}$), and she doesn't like dark and suspenseful thrillers ($q^{th}$ feature--low value for $U\_{iq}$), however, she likes the movie Eyes Wide Shut (even though it has a high value for $V\_{jq}$), perhaps because an underlying reason that makes her not like dark suspenseful movies perhaps disappears if Tom Cruise is in the movie. For a model to capture such a pattern, it has to allow some sort of **nonlinear cross feature interactions** between features $p$ and $q$.  
$$=> Nonlinear\ Model: MAE (CV) = 0.6150$$
```
R_pred = np.dot(U,V) + alpha1*(xft) + alpha2*(uv_sq)
```

- The computational price paid for a mere 1% increas in mae rate is that the runtime for one epoch went from $31$ sec for pure linear model to $60$ sec when incorporating all 3 types of nonlinear feature crossings:  
<p align="center">($U\_{ip}$ ~ $V\_{jq}$),&nbsp; ($U\_{ip}$ ~ $U\_{iq}$),&nbsp; ($V\_{jp}$ ~ $V\_{jq}$)</p>


## 7. Improvements on the Algorithm
1. The regularization term needs to take into account the average rating for each user $U\_i$, $\mu_{U_i}$. 

 









