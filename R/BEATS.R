#' Bayesian joint modeling of single-cell expression data and bulk spatial transcriptomic data
#'
#' The function BEATS implements a Bayesian method for integrating scRNA-seq data and bulk ST data. BEATS cluster cells, partition spatial spots into different regions, and obtain cellular enrichments of spots simultaneously. It employs a hybrid Markov chain Monte Carlo algorithm to perform posterior inference.
#' 
#' @param scRNA_data_matr The scRNA-seq data matrix, where rows represent genes and columns represent cells. Matrix values need to be non-negative and continuous after adjusting for cells’ library sizes. Please do not take logarithm during data preprocessing.
#' @param ST_data_matr The bulk spatial transcriptomic (ST) data matrix, where rows represent genes and columns represent spots. Matrix values need to be normalized first. Noteworthy, the column order of spots has to be consistent with the order of spots in the argument spot_matr. For example, the coordinates of spot 1 are the first row of matrix spot_matr, the coordinates of spot 2 are the second row of matrix spot_matr, and so on.
#' @param spot_matr The coordinate matrix of ST data, where the first column represents row indices of spots and the second column represents column indices of spots.
#' @param n_celltype An integer, denoting the number of cell types.
#' @param n_region An integer, denoting the number of regions.
#' @param warm_cluster_label_init The initialization of cluster labels is random or based on k-means. The default is FALSE, corresponding to the random initialization.
#' @param num_iterations The number of Gibbs sampler iterations. The default is 10000.
#' @param num_burnin The number of iterations in burn-in, after which the posterior samples are used to estimate the unknown parameters. The default is the first half of total iterations.
#' @param collect_post_sample Logical, collect the posterior samples or not. If users are only interested in the estimates, set it as FALSE to save the memory. If users would like to use posterior samples to construct credible intervals or for other uses, set it as TRUE. The default is FALSE.
#' @param hyperparameters A vector, which indicates 20 hyper-parameters in the priors or proposal distributions. The first two elements are the mean and standard deviation of the normal prior distribution for \eqn{\mu} respectively. The third and fourth elements represent the shape and rate of inverse-gamma prior distribution for cell-type expression variance. The fifth element is the sum of all the concentration parameters of the Dirichlet proposal distribution for vector \eqn{p_s} where s is the region number. The sixth and seventh elements correspond to the shape and rate of inverse-gamma prior distribution for error variance respectively, while the eighth and ninth elements are the shape and rate of inverse-gamma prior distribution for bulk expression data expression variance respectively. The tenth element stands for the standard deviation of the normal proposal distribution for \eqn{\psi}. The eleventh and twelfth elements represent the mean and standard deviation of the normal prior distribution for \eqn{\psi} respectively. The thirteenth and fourteenth elements correspond to the step size and step number of leapfrog iteration for \eqn{\theta} respectively. The fifteenth and sixteenth elements are, respectively, the means of the normal priors for \eqn{\lambda_0} and \eqn{\lambda_1}. The seventeenth and eighteenth elements correspond to the variances of the normal priors for \eqn{\lambda_0} and \eqn{\lambda_1} respectively. The last two elements are the step size and step number of leapfrog iteration for \eqn{\lambda} respectively. The default is 0, 5, 2, 0.1, 1000, 3, 0.1, 5, 0.01, 1, 0, 10, 0.2, 20, 2, -2, 0.25, 0.25, 0.01, 10.
#' @param hyperparameters_conc A list, the first element is a vector of n_celltype dimension, describing the parameters of Dirichlet prior for \eqn{\pi}. The default is a vector filled with two. The latter one is a matrix with n_celltype rows and n_region columns, the s-th column vector is the parameters of Dirichlet prior for \eqn{p_s}. The default is a matrix with each element being three.
#' @param print_label Logical, whether or not to print summarized cell type label and spot region label information after each iteration. The default is FALSE.
#' @param print_per_iteration Integer, how many iterations to print the iteration information when print_label is FALSE. The default is 1000.
#' 
#' @return BEATS returns an R list including the following information.
#' \item{cell_labels}{A vector, indicating the estimated cell types for each cell.}
#' \item{region_labels}{A vector, indicating the estimated region types for each spot.}
#' \item{cell_type_mean_expr}{A matrix of cell type expression profiles, in which rows are genes and columns correspond to cell types.}
#' \item{prop}{A matrix of cell type proportions in each region, in which rows are cell types and columns are regions. Specifically, prop[k, s] is the proportion of cell type k in region s.}
#' \item{region_mean_expr}{A matrix of region expression profiles, in which rows are genes and columns correspond to region types.}
#' \item{lam0}{A vector, the estimated \eqn{\lambda_0} for each gene.}
#' \item{lam1}{A vector, the estimated \eqn{\lambda_1} for each gene.}
#' \item{region_interaction}{A symmetric matrix measures the interaction effects between different types of regions.}
#' \item{cell_label_post}{Collected posterior samples of cell_labels when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{region_label_post}{Collected posterior samples of region_labels when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{region_interaction_post}{Collected posterior samples of region_interaction when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{prop_post}{Collected posterior samples of prop when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{cell_type_mean_expr_post}{Collected sampls of cell_type_mean_expr when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{lam0_post}{Collected posterior samples of lam0 when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{lam1_post}{Collected posterior samples of lam1 when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{sgm_sq_y_post}{Collected posterior samples of cell-type expression variance when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{sgm_sq_h_post}{Collected posterior samples of error variance when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{sgm_sq_z_post}{Collected posterior samples of bulk expression data expression variance when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{theta_post}{Collected posterior samples of \eqn{\theta} when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{region_mean_expr_post}{Collected posterior samples of region_mean_expr when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' 
#' @examples 
#' #Data generation
#' set.seed(20200202)
#' #cell number
#' n <- 100
#' #gene number
#' G <- 50
#' #number cell types
#' K <- 2
#' #number of region states
#' S <- 3
#' #length
#' L <- 10
#' #width
#' W <- 10
#' #spot_matr
#' ind_mat <- NULL
#' for (j in 1:10) {
#'   for(i in 1:10){
#'     ind_mat <- rbind(ind_mat, c(i, j))
#'   }
#' }
#' spot_matr <- as.matrix(ind_mat)
#' #region indicators
#' R <- matrix(1, 10, 10)
#' R[1:5,1:5] <- 2
#' R[1:5,6:10] <- 3
#' R <- c(R)
#' #spot number
#' m <- nrow(spot_matr)
#' #cell cluster proportion
#' pi_K <- c(0.6, 0.4)
#' #cell type vector
#' C <- NULL
#' for (k in 1:K) {
#'   C <- c(C, rep(k, n*pi_K[k])) 
#' }
#' #cell-type-specific mean
#' NDE_mu <- rgamma(G,shape = 4,scale = 0.5)
#' mu_1 <- NDE_mu
#' mu_2 <- NDE_mu
#' mu_2[1:(5*G/10)] <- rnorm(5*G/10, mean=rep(c(4,5), each = 2.5*G/10), sd=0.2)
#' mu <- cbind(mu_1, mu_2)
#' #cell type proportions in each region
#' p <- t(rbind(c(0.7, 0.3),
#'              c(0.4, 0.6),
#'              c(0.2, 0.8)))
#' #region effects
#' h <- matrix(NA, G, S)
#' for(s in 1:S){
#'   h[,s] <- rnorm(G, rowSums(sweep(mu, 2, p[,s], "*")), 0.05)
#' }
#' 
#' #observed expression data 
#' #scRNA-seq
#' theta <- matrix(NA, G, n)
#' for(i in 1:n){
#'   theta[ ,i] <- rnorm(G, mean = mu[,C[i]], sd = 0.2)
#' }
#' lambda0 <- rnorm(G, 1, 0.1) 
#' lambda1 <- rnorm(G, -1, 0.1) 
#' X <- exp(theta)
#' tmp <- pnorm(lambda0 + lambda1 * theta)
#' r <- matrix(runif(G*n), G, n)
#' X[r <= tmp] <- 0
#' scRNA_data_matr <- X
#' 
#' #ST data
#' Z <- matrix(NA, G, m)    
#' for(i in 1:m){
#'   Z[ ,i] <- rnorm(G, mean = h[ ,R[i]], sd = 0.2)
#' }
#' ST_data_matr <- Z
#' 
#' #run BayesEATS
#' library(BayesEATS)
#' t1 <- Sys.time()
#' Result <- BEATS(scRNA_data_matr, ST_data_matr, spot_matr, n_celltype = 2,
#'                 n_region = 3, num_iterations = 5000, print_per_iteration = 500)
#' t2 <- Sys.time()
#' 
#' #time cost
#' print(t2 - t1)
#' 
#' #Compared with true cell type labels
#' table(Result$cell_labels, C)
#' 
#' #Compared with true region labels
#' table(Result$region_labels, R)
#' 
#' #Estimate for the cellular composition matrix
#' cell_comp <- Result$prop
#' rownames(cell_comp) <- c("cell type 1", "cell type 2")
#' colnames(cell_comp) <- c("region 1", "region 2", "region 3")
#' cell_comp
#' 
#' @references 
#' @export
BEATS <- function(scRNA_data_matr, ST_data_matr, spot_matr, n_celltype, n_region, warm_cluster_label_init = FALSE, 
                  num_iterations = 10000, num_burnin = floor(num_iterations/2), collect_post_sample = FALSE, 
                  hyperparameters = c(0, 5, 2, 0.1, 1000, 3, 0.1, 5, 0.01, 1, 0, 10, 0.2, 20, 2, -2, 0.25, 0.25, 0.01, 10), 
                  hyperparameters_conc = list(rep(2, n_celltype), matrix(3, n_celltype, n_region)),
                  print_label = FALSE, print_per_iteration = 1000) {
  
  scRNA_data_matr <- as.matrix(scRNA_data_matr)
  ST_data_matr <- as.matrix(ST_data_matr)
  spot_matr <- as.matrix(spot_matr)
  
  #zero index
  ind_zero <- (scRNA_data_matr == 0)
  
  if ((sum(rowMeans(ind_zero) == 1)) > 0 | sum(rowMeans(ST_data_matr == 0) == 1) > 0) {
    stop("Please remove the genes having zero expression across spots or cells!")
  }
  
  G <- dim(scRNA_data_matr)[1]
  n <- dim(scRNA_data_matr)[2]
  m <- dim(ST_data_matr)[2]
  
  Is_nonneighbor <- function(ind_mat){
    non_nei <- NULL
    for(i in 1:nrow(ind_mat)){
      l <- ind_mat[i, 1]
      w <- ind_mat[i, 2]
      nei_tmp <- matrix(c(l,w+1,l,w-1,l+1,w,l-1,w), 4, 2, byrow = T)
      tmp1 <- rep(0, 4)
      for(j in 1:4){
        tmp2 <- rep(0, nrow(ind_mat))
        for(nr in 1:nrow(ind_mat)){
          tmp3 <- nei_tmp[j, ] - ind_mat[nr,]
          if(sum(abs(tmp3)) != 0){
            tmp2[nr] <- 1
          }
        }
        if(sum(tmp2) == nrow(ind_mat)){
          tmp1[j] <- 1
        }
      }
      
      if(sum(tmp1) == 4){
        # record coordinate of spots do not have neighbor
        non_nei <- rbind(non_nei, c(l,w))
      }
    }
    return(non_nei)
  }
  
  non_nei <- Is_nonneighbor(spot_matr)

  if(length(non_nei) > 0){
    print("Spots below do not have neighbors")
    print(non_nei)
    stop("Please remove spots that do not have a neighbor")
  }
  
  #initialize theta_t
  theta_t <- scRNA_data_matr
  for (g in 1:G) {
    theta_t[g, ind_zero[g, ]] <- quantile(scRNA_data_matr[g,!ind_zero[g,]], probs = 0.05)
  }
  theta_t <- log(theta_t)
  
  #initialize types
  if (warm_cluster_label_init == FALSE) {
    C_t <- sample(1:n_celltype, n, replace = TRUE)
    R_t <- sample(1:n_region, m, replace = TRUE)
  } else {
    C_t <- kmeans(t(log2(scRNA_data_matr+1)), n_celltype)$cluster
    R_t <- kmeans(t(ST_data_matr), n_region)$cluster
  }
  
  #initialize p_t
  p_t <- matrix(1/n_celltype, n_celltype, n_region)
  
  #initialize mu_t
  mu_t <- NULL
  for (k in 1:n_celltype) {
    mu_t <- cbind(mu_t, rep(k,G))
  }
  
  #initialize h_t
  h_t <- matrix(NA, G, n_region)
  for(s in 1:n_region){
    h_t[ ,s] <- rowMeans(ST_data_matr[ , R_t == s])
  }
  
  #initialize sgm
  sgm_sq_t <- apply(theta_t - mu_t[, C_t], 1, var)
  sgm_sq_star_t <- apply(ST_data_matr - h_t[ ,R_t], 1, var)
  sgm_sq_vare_t <- var(c(h_t))
  
  #initialize psi_t
  psi_t <- matrix(rnorm(n_region*n_region, hyperparameters[11], 0.01),n_region, n_region)
  psi_t <- (psi_t + t(psi_t))/2
  diag(psi_t) <- 0
  
  #initialize lambda_t
  lambda0_t <- rnorm(G, mean=hyperparameters[15], sd=sqrt(hyperparameters[17]))
  lambda1_t <- rnorm(G, mean=hyperparameters[16], sd=sqrt(hyperparameters[18]))
  
  
  gam_1 <- hyperparameters_conc[[1]]
  gam_2 <- hyperparameters_conc[[2]]
  
  ###############################################################
  #######################  Gibbs Sampler ########################
  ###############################################################
  num_saved <- num_iterations - num_burnin
  
  Result <- MCMC_full(num_iterations, num_saved, ST_data_matr, theta_t, ind_zero,
                      mu_t, p_t, h_t, psi_t, spot_matr,
                      lambda0_t, lambda1_t, C_t, R_t,
                      sgm_sq_t, sgm_sq_star_t, gam_1, gam_2,
                      G, n, m, n_celltype, n_region,
                      sgm_sq_vare_t, eta_mu = hyperparameters[1], tau_mu = hyperparameters[2],
                      alpha_1 = hyperparameters[3], beta_1 = hyperparameters[4], gamma_0 = hyperparameters[5],
                      alpha_vare = hyperparameters[6], beta_vare = hyperparameters[7],
                      alpha_2 = hyperparameters[8], beta_2 = hyperparameters[9],
                      tau_0 = hyperparameters[10], eta_psi = hyperparameters[11], tau_psi = hyperparameters[12],
                      epsilon_theta = hyperparameters[13], num_step_theta = hyperparameters[14],
                      lam0_0 = hyperparameters[15], lam1_0 = hyperparameters[16], sigma2_lam0 = hyperparameters[17],
                      sigma2_lam1 = hyperparameters[18], epsilon_lam = hyperparameters[19], num_step_lam = hyperparameters[20], 
                      iter_save = collect_post_sample, iter_print = print_per_iteration, class_print = print_label)
  return(Result)
}
