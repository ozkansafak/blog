---
date: 2017-11-18T15:49:35-08:00
draft: false
title: "A Matrix Factorization Model on tensorflow (with Nonlinear Cross Features)"
markup: "markdown"
author: "Safak Ozkan"
---

---

## 1. Problem Description
We are given a rating matrix $R$ where only a small fraction of the entries are provided; otherwise the rest of them are missing. The task of a Recommender System is to predict those missing entries. As in most Machine Learning problems the assumption here is that there's an underlying pattern of how users rate movies.

By the nature of the problem, $R$ is a sparse matrix, where the sparsity comes not from zero entries but from empty records. Therefor, we represent the data in the rating matrix $R$ in 3 columns: $i$: user ID , $j$: movie ID and $R_{ij}$: the rating (see Table 1).    
   

<font size="+1"><strong><p align="center">Table 1. A conceptual sketch of the Ratings in ml-20m data in sparse format </p></strong></font>


| $i$: user ID | $j$: movie ID   | $R_{ij}$: the rating |
|:-----:|:---------:|:-----:|
| 0      |   14     | 3.5   |
| 0      |  7305    | 4.0  |
| 0      |  16336   | 3.5 |
| .      |   .      |  .  |
| .      |   .      |  .  |
| 1      |   52     | 4.0 |
| 1      |   986    |  4.0  |
| 1      |   1455   |  3.5 |
| 1      |   1705   |  5.0    |
| 1      |   5598   |  4.0    |
| .      |     .    |  .    |
| .      |     .    |  .    |
| 138493 |   27278  |  5.0   |

---

## 2. MovieLens 20M dataset

- [MovieLens dataset](https://grouplens.org/datasets/movielens/20m/) data set consists of 20,000,263 ratings from 138,493 users on 27,278 movies. Ratings are provided for only $0.5\%$ of all possible entries in $R$.
- All ratings are given in the interval [0.5, 5.0] with increments of 0.5:  
{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}
- Since the input data was ordered according to user IDs, it was crucial to shuffle the data before splitting it into training, CV and test sets.
- The Input Data is split accordingly:
	- $64\%$ -- training data,
	- $16\%$ -- cross validation data,
	- $20\%$ -- test data.
- We abstain from imposing an explicit **bias term** in the feature vectors $U$ and $V$. In the Matrix Factorization scheme, the embeddings are free to learn biases if necessary.
- Since no particular bounds are imposed on the entries in the feature vectors $U$ and $V$, the model is free to learn positive or negative real numbers as feature coefficients.  

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
<font size="+1"><b><p align="center">Figure 1. Histogram of (a) all ratings in ml-20m data (b) mean of ratings per user \(c) mean of ratings per movie, and (d) 
	number of ratings provided by users. Minimum number of ratings provided by a user is 20, and maximum is 9254 ratings.
</p></b></font>


## 3. Matrix Factorization Model
The terms  *Matrix Factorization*, *Low-Rank Matrix Factorization* and *Collaborative Filtering*  all refer to the same recommender system model in the context of the current problem.  In essence, this model is based on the assumption that users who liked the same movies are likely to feel similarly towards other movies.  The term *collaborative* refers to the observation that when a large set of users are involved in rating the movies, these users effectively collaborate to help the model  better predict the movie ratings for everyone because every new rating will help the algorithm learn better features for the complete user-movie system.   

The Collaborative Filtering Model can also be described as reconstructing a **low rank approximation** of matrix $R$ via its **Singular Value Decomposition** $R = U \cdot \Sigma \cdot V^T$. The low-rank reconstruction is achieved by only retaining the largest $k$ singular values, $R\_k = U \cdot \Sigma\_k \cdot V^T$.

**Eckart-Young Theorem** states that if $R_k$ is the best rank-$k$ approximation of $R$, then it's necessary that:  
 
&emsp;&emsp;&emsp;   1. $R\_k$ minimizes the Frobenius norm $||R - R\_k||\_F^2$ and   
&emsp;&emsp;&emsp;   2. $R\_k$ can be constructed by retaining only the largest $k$ singular values in the diagonal matrix $\Sigma\_k$ of the SVD formulation.

We can further absorb the diagonal matrix $\Sigma\_k$ into $U$ and $V$ and express the factorization as a simple dot product between the feature matrices for users and movies.
 
<p align="center">$\hat{R}_{k(m \times n)} = U_{(m \times k)}^\ \cdot V_{(k \times n)}^T$</p>

where, the parentheses indicate matrix size.  
$m$: number of users ($m = 138493$)  
$n$: number of movies ($n = 27278$)  
$k$: rank hyperparameter (typically $k \approx 10$).  
$U$: user feature matrix  
$V$: movies feature matrix    

Hence, we can formulate the problem as an **optimization problem** minimizing the following loss function $L$ via SGD.         

$$argmin_{\ U,V}\ L = ||R - \hat{R}||_F^2$$

It's important to note that the Frobenius norm is computed only as a **partial summation** over the entries in $\hat{R}$ where a rating is provided---or equivalently over the list of ratings as shown in Table 1. The optimization procedure searches for the values of all entries in $U$ and $V$. There are $(m+n) \times k$ many such  tunable variables.   

The hyperparameter $k$ is to be chosen carefully by cross-validation. Too small a $k$ value would not be enough to explain the pattern in the data adequately (*underfitting*), and too large a $k$ value would result in a model fitting on the random noise over the pattern (*overfitting*).

It's worth making a brief interpretation of the feature matrices $U$ and $V$. In the $k$-rank approximation scheme, each rating $R\_{ij}$ is expressed as the dot product $U\_i^\ \cdot V\_j^T$ as shown in Figure 2. The goal of our optimization routine is for the model to learn a **latent feature vector** (or alternatively an **embedding vector**) for each user and movie.  The term *latent* implies that the features are not explicitly defined as a part of the model nor they can be interpreted definitively once the embeddings are learned. Each entry in $U_i$ and $V_j$ corresponds to the weight coefficient of an abstract feature. These features can specify the genre of the movie or how much action or drama contained in the movie or any other distinguishing quality that would help characterize how the users rate movies. Hence, the dot product representation of the ratings $R\_{ij} = U\_i^\ \cdot V\_j^T$ expresses a **linear combination** of   

&emsp;&emsp;&emsp; 1. how much that feature is favored by the user-$i$, and   
&emsp;&emsp;&emsp; 2. how much that feature is contained in the movie-$j$.      

<img src="/R=UV^T.png" alt="R=UV^T" width="1000" />
<font size="+1"><p align="center"><b>Figure 2. A conceptual sketch of the ratings matrix $R$ decomposed into its factors: user and movie feature matrices, $U$ and $V$. Dots in the figure "$\cdot$" illustrate rating values that are provided by users; and question marks "$?$" the missing values. Each entry $R\_{ij}$ is expressed as a dot product of the user and movie embedding vectors $U\_i$ and $V\_j$, respectively.</b></p></font>  

---   

## 4. Linear, Nonlinear and Cross Features

##### Linear Terms:
-  Standard collaborative filtering model consists of the following linear term:
$$\hat{R}\_{ij} = (R\_{lin})\_{ij} = U\_{i}^\ \cdot V\_{j}^T$$
In this dot product, $p^{th}$ feature coefficient of $U\_{i}$ is multiplied with the corresponding $p^{th}$ coefficient of $V\_{j}$. The contributions from each feature are added up into a total sum.

- Subsequently, I added a sigmoid filter $R\_{ij} = 4.5 \cdot \sigma(U\_i^\ \cdot V\_j^T) + 0.5$ which forced all output predictions into the interval [0.5, 5.0], where $\sigma$ is the sigmoid function. This small detail helped lower the Mean Absolute Error (MAE) Rate from approximately `.64` to `.62`. The reason for this is that in the absence of sigmoid activation, some predictions fall outside the natural range $[0.5, 5.0]$. The sigmoid activation squashes the predictions to the correct range and hence closer to their actual values.   
\
The MAE on the cross validation set for the pure linear model, $\hat{R} = R\_{lin}$, is:
$$=> Linear\ Model: MAE (CV) = 0.622$$  

##### Nonlinear Terms:
- I experimented with adding a 2<sup>nd</sup> order term to the rating model:    
$$(R\_{nl})\_{ij} = \sum\_{p=1}^k \Big[U\_{ip} V\_{jp}\Big]^2$$

- However, the quadratic terms didn't result in any discernible reduction in the final error rate and it was not used in the final model.  
$$+\sum\_{p=1}^{k}\sum\_{q=p+1}^{k} \Big[(X\_{UU})\_{pq}\ U\_{ip} U\_{jq}+ (X\_{VV})\_{pq}\ V\_{ip}  V\_{jq}\Big]$$

##### Cross Feature Terms:
- Cross feature terms introduce the following 2<sup>nd</sup> order nonlinearity to the rating model:  
$$(R\_{xft})\_{ij} = \sum\_{p=1}^{k}\sum\_{q=1}^{k} (X\_{UV})\_{pq}\ U\_{ip} V\_{jq}$$
where,  
$(X\_{UV})\_{pq}$: user-movie cross feature coefficient between features $p$ and $q$,   
\
$X\_{UV}$ is of size $k \times k$. Its components are to be learned by the optimization routine. In the above term, feature-$p$ of $U\_i$ gets multiplied by feature-$q$ of $V\_j$ and the contribution to the rating $R\_{ij}$ is controlled by the cross feature coefficient $(X\_{UV})\_{pq}$.  
\
- The interpretation of the cross feature term could be made as follows: if a user likes the actor Tom Cruise (a large value for $U\_{ip}$), but she doesn't like dark suspenseful movies (a small value for $U\_{iq}$), however, she likes the movie Eyes Wide Shut (even though it has a high value for $V\_{jq}$), because an underlying reason that makes her not like dark suspenseful movies perhaps vanishes if Tom Cruise is in the movie. For a model to capture such a pattern, it has to allow some sort of **nonlinear cross feature interactions** between features $p$ and $q$.   

- However, the gain in the addition of cross feature terms over the linear terms on the MAE Rate is a mere $1.5\%$:
$$\hat{R} = R\_{lin} + R\_{xft}$$
$$=> Nonlinear\ Cross\ Feature\ Model: MAE (CV) = 0.612$$ 

- The computational price paid for a mere $2\%$ improvement in MAE Rate is that the runtime increased from $25\ sec/epoch$ to $38\ sec/epoch$ when incorporating the cross feature interaction  ($U\_{ip}$ ~ $V\_{jq}$).

- > I also experimented on adding two more cross feature terms as follows:  $(X\_{UU})\_{pq}\ U\_{ip} U\_{jq} + (X\_{VV})\_{pq}\ V\_{ip}  V\_{jq}$. Not surprisingly, this didn't produce any improvement in the final MAE Rate only caused a tiny bit of overfitting. The cross feature interaction between $U\_p$---$U\_q$ and $V\_p$---$V\_q$ can be learned by increasing $k$, as well.
\
- > I also experimented on implementing a $k \times k$ separate cross feature tensor of size for each user. This produces a 3-dimensional tensor $X\_{UV}$ of size $m \times k \times k=$ 13,849,300 new tunable parameters instead of only $k \times k$ which is a mere 100.



---

## 5. Regularization
An $L\_2$ regularization term applied naively on the feature vectors $U$ and $V$ would penalize all nonzero components. This would encourage the coefficients in $U$ and $V$ to be close to zero.  However, we would rather have the coefficients in $U$ and $V$ regress towards the mean rating of the corresponding user (or alternatively corresponding movie) instead of zero. And for the nonlinear cross feature terms I kept the regularization term naive to regress towards the value zero. 

In essence, we are imposing a penalty for any behavior that diverges from the average pattern.  In this spirit, I formulated the $L\_2$ regularization term as follows:


$$\Omega\_{lin} = \sum\_{i=1}^m\ \Big(U_i - \mu\_{U_i} \Big)^2 +\sum\_{j=1}^n\ \Big(V_j - \mu\_{V_j}\ \Big)^2$$
$$\Omega\_{xft} = \sum\_{p=1}^k \sum\_{q=1}^k\ (X\_{UV})\_{pq}$$

where,  
$\Omega\_{linear}$: regularization term for linear features   
$\Omega\_{xft}$: regularization term for cross features   
$\mu\_{U_i}$: mean rating for user-$i$    
$\mu\_{V_j}$: mean rating for movie-$j$   


```python
reg = (tf.reduce_sum((stacked_U - stacked_u_mean)**2) + 
       tf.reduce_sum((stacked_V - stacked_v_mean)**2) + 
       tf.reduce_sum(X_UV**2)) / (BATCH_SIZE*k)
```

However, regularization didn't improve the MAE Rate because of the over abundance of data which is already the best implicit regularizer than any other explicitly imposed regularization term.

---

## 6. Practical Methodology and Developing the Model on `tensorflow`
##### Shape of input tensors `R` and `R_indices`:
- A particular challenge in implementing a Matrix Factorization algorithm on `tensorflow` is that we can't pass `None` for the `shape` argument while declaring the input data tensors `R` and `R_indices` as in `R = tf.placeholder(..., shape=(None))`.  The `shape` parameter corresponds to the number of ratings a single batch contains. To make the SGD work, I had to fix the `shape` of the `tf.placeholder` variables `R` and `R_indices`  to `shape=(BATCH_SIZE, k)`.  This is a small price to pay which allowed me to build the collaborative filtering model on `tensorflow` and use GPU computation and symbolic differentiation. It was also easier to experiment with additional nonlinear terms in the loss function without having the worry about computing the partial differentials by hand. 

```python
R = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,))
R_indices = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE,2))
u_mean = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,1)) 
v_mean = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE,1)) 
```

- At each SGD step a mini-batch of ratings $R_{ij}$ and the corresponding user-movie index pairs $(i,j)$ are fed into the computational graph. In order to achieve this, we have to stack the corresponding embedding vectors into 2-D tensors `U_stack` and `V_stack` where both `U_stack.getshape()` and `V_stack.getshape()` equal to `(BATCH_SIZE,k)`.   
\
The implementation of stacking tensors  on`tensorflow` is a little trickier than in `numpy`:

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

##### Initialization: 
- The feature vectors $U$ and $V$ are initialized by sampling from a Gaussian Distribution with mean $\mu = \sqrt{ 3.5/k}$, and standard deviation $\sigma = 0.2$. For the cross feature vectors $X\_{UV}$, $X\_{UV}$ and $X\_{VV}$, the mean was chosen empirically by experimentation as $\mu = -1/k$ and standard deviation $\sigma = 0.2$.

---

## Github Repo
- Enjoy:
\
https://github.com/ozkansafak/Matrix_Factorization

---

## References:
1. Learning From Data, Yaser S. Abu-Mostafa, Malik Magdon-Ismail, Hsuan-Tien Lin, 2012.
2. Machine Learning, Andrew Ng, Coursera online class, 2011.
3. Deep Learning, Ian Goodfellow, Yoshua Bengio, Aaron Courville, 2016.






