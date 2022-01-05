## BayesEATS: an R package for Bayesian joint modeling of single-cell expression data and bulk spatial transcriptomic data

The R package BayesEATS implements the method BEATS proposed by Jinge Yu, Qiuyu Wu and Xiangyu Luo (2020+) that integrates scRNA-seq data and bulk ST data to simultaneously cluster cells, partition spatial spots into different regions, and estimate cellular enrichments of spots in the Bayesian framework. In this package, we employed a hybrid Markov chain Monte Carlo algorithm to perform efficient posterior inference for BEATS model. BayesEATS can be installed in commonly used operating systems including Windows, Linux and Mac OS. 

Notice that there is a typo in equation (2) in the paper, $\pi_k$ was missing. However, there is **no problem** in the derivation of the latter algorithm, and the typo **does not affect** the simulation and real data results. The correct form of equation (2) in the paper is:

![image](https://github.com/jingeyu/BayesEATS/tree/master/images/revise.png)


## Prerequisites and Installation

1. R version >= 3.6.
2. R packages: Rcpp (>= 1.0.3), RcppArmadillo (>= 0.9.850.1.0).
3. Install the package BayesEATS.

```
devtools::install_github("jingeyu/BayesEATS")
```


## Example Code
Following shows an example that generates data and runs the main function "BEATS" in our package. 

``` {r, eval=FALSE}
library(BayesEATS)
#############################################
#Data generation
#############################################
set.seed(20200202)
#cell number
n <- 100
#gene number
G <- 50
#number cell types
K <- 2
#number of region states
S <- 3
#length
L <- 10
#width
W <- 10
#spot_matr
ind_mat <- NULL
for (j in 1:10) {
  for(i in 1:10){
    ind_mat <- rbind(ind_mat, c(i, j))
  }
}
spot_matr <- as.matrix(ind_mat)
#region indicators
R <- matrix(1, 10, 10)
R[1:5,1:5] <- 2
R[1:5,6:10] <- 3
R <- c(R)
#spot number
m <- nrow(spot_matr)
#cell cluster proportion
pi_K <- c(0.6, 0.4)
#cell type vector
C <- NULL
for (k in 1:K) {
  C <- c(C, rep(k, n*pi_K[k])) 
}
#cell-type-specific mean
NDE_mu <- rgamma(G,shape = 4,scale = 0.5)
mu_1 <- NDE_mu
mu_2 <- NDE_mu
mu_2[1:(5*G/10)] <- rnorm(5*G/10, mean=rep(c(4,5), each = 2.5*G/10), sd=0.2)
mu <- cbind(mu_1, mu_2)
#cell type proportions in each region
p <- t(rbind(c(0.7, 0.3),
             c(0.4, 0.6),
             c(0.2, 0.8)))
#region effects
h <- matrix(NA, G, S)
for(s in 1:S){
  h[,s] <- rnorm(G, rowSums(sweep(mu, 2, p[,s], "*")), 0.05)
}

#observed expression data 
#scRNA-seq
theta <- matrix(NA, G, n)
for(i in 1:n){
  theta[ ,i] <- rnorm(G, mean = mu[,C[i]], sd = 0.2)
}
lambda0 <- rnorm(G, 1, 0.1) 
lambda1 <- rnorm(G, -1, 0.1) 
X <- exp(theta)
tmp <- pnorm(lambda0 + lambda1 * theta)
r <- matrix(runif(G*n), G, n)
X[r <= tmp] <- 0
scRNA_data_matr <- X

#ST data
Z <- matrix(NA, G, m)    
for(i in 1:m){
  Z[ ,i] <- rnorm(G, mean = h[ ,R[i]], sd = 0.2)
}
ST_data_matr <- Z

#############################################
#run the main function "BEATS"
#############################################
t1 <- Sys.time()
Result <- BEATS(scRNA_data_matr, ST_data_matr, spot_matr, n_celltype = 2,
                n_region = 3, num_iterations = 5000, print_per_iteration = 500)
t2 <- Sys.time()

#time cost
print(t2 - t1)

#Compared with true cell type labels
table(Result$cell_labels, C)

#Compared with true region labels
table(Result$region_labels, R)

#Estimate for the cellular composition matrix
cell_comp <- Result$prop
rownames(cell_comp) <- c("cell type 1", "cell type 2")
colnames(cell_comp) <- c("region 1", "region 2", "region 3")
cell_comp

```
 
or you can simply run
``` {r, eval=FALSE}
library(BayesEATS)
example(BEATS)
```

## Remarks
* If you have any questions regarding this package, please contact Jinge Yu at yjgruc@ruc.edu.cn or Qiuyu Wu at w.qy@ruc.edu.cn.

