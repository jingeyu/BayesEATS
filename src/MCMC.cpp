#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <Rmath.h>

using namespace arma;
using namespace Rcpp;
using namespace std;

// [[Rcpp::depends("RcppArmadillo")]]

arma::vec U_theta(const arma::vec &theta_miss, const arma::vec &mu, const arma::vec &sigma_sq, const arma::vec &lambda0_miss, const arma::vec &lambda1_miss) {
  arma::vec U = (theta_miss - mu) %  (theta_miss - mu) / 2.0 / sigma_sq -
    log(arma::normcdf(lambda0_miss + lambda1_miss % theta_miss));
  return U;
}

void U_theta_grad(arma::vec &U_grad, const arma::vec &theta_miss, const arma::vec &mu, const arma::vec &sigma_sq, const arma::vec &lambda0_miss, const arma::vec &lambda1_miss) {
  U_grad = (theta_miss - mu) / sigma_sq - 
    (lambda1_miss % normpdf(lambda0_miss + lambda1_miss % theta_miss)) / arma::normcdf(lambda0_miss + lambda1_miss % theta_miss);
}

void HMC_theta(arma::vec &theta_new, arma::vec &log_r, const arma::vec &theta_miss, const arma::vec &mu, const arma::vec &sigma_sq, const arma::vec &lambda0_miss,
               const arma::vec &lambda1_miss, const arma::vec &p_rnorm, const double &epsilon, const int &num_step) {
  //The kinetic energy has the simplest form arma::sum(p^2)/2
  arma::vec U_grad;
  
  theta_new = theta_miss;
  arma::vec p_new = p_rnorm;
  
  // a half step for momentum at the beginning
  U_theta_grad(U_grad, theta_new, mu, sigma_sq, lambda0_miss, lambda1_miss);
  p_new -= epsilon *  U_grad / 2.0;
  
  // full steps for position and momentum
  for (int i = 0; i < (num_step-1); i++) {
    theta_new += epsilon * p_new;
    
    U_theta_grad(U_grad, theta_new, mu, sigma_sq, lambda0_miss, lambda1_miss);
    p_new -= epsilon * U_grad;
  }
  theta_new += epsilon * p_new;
  
  U_theta_grad(U_grad, theta_new, mu, sigma_sq, lambda0_miss, lambda1_miss);
  p_new -= epsilon * U_grad / 2.0;
  
  p_new = -p_new;
  log_r = U_theta(theta_miss, mu, sigma_sq, lambda0_miss, lambda1_miss) - U_theta(theta_new, mu, sigma_sq, lambda0_miss, lambda1_miss) +
    (p_rnorm % p_rnorm) / 2.0 - (p_new % p_new) / 2.0;
}


void theta_update(arma::mat &theta_t, const arma::mat &ind_zero, const arma::mat &mu_t, const arma::vec &sgm_sq_t, const arma::vec &lambda0_t, 
                  const arma::vec &lambda1_t, const arma::vec &C_t, const int &n, const double &epsilon, 
                  const int &num_step) {
  for (int i = 0; i < n; i++) {
    arma::vec ind_zero_i = ind_zero.col(i);
    arma::uvec ind_0 = arma::find(ind_zero_i == true);
    
    if (ind_0.n_elem > 0) {
      arma::vec theta_i = theta_t.col(i);
      arma::vec theta_i_0 = theta_i(ind_0);
      
      arma::vec mu_i = mu_t.col(C_t(i)-1);
      
      arma::vec p_rnorm = arma::randn(ind_0.n_elem);
      arma::vec tmp_unif = arma::randu(ind_0.n_elem);
      
      arma::vec theta_star_0;
      arma::vec log_r;
      HMC_theta(theta_star_0, log_r, theta_i_0, mu_i(ind_0), sgm_sq_t(ind_0), lambda0_t(ind_0), lambda1_t(ind_0), p_rnorm, epsilon, num_step);
      
      arma::uvec ind = arma::find(tmp_unif < exp(log_r));
      theta_i_0(ind) = theta_star_0(ind);
      arma::uvec i_vec(1);
      i_vec(0) = i;
      theta_t.submat(ind_0, i_vec) = theta_i_0;
    }
  }
}


double U_lam(const arma::vec &lambda, const arma::vec &theta_miss, const arma::vec &theta_obs, 
             const double &lam0_0, const double &lam1_0, const double &sigma2_lam0, const double &sigma2_lam1) {
  arma::vec U = - arma::sum(log(1.0 - arma::normcdf(lambda(0) + lambda(1) * theta_obs))) - arma::sum(log(arma::normcdf(lambda(0) + lambda(1) * theta_miss))) +
    (lambda(0) - lam0_0) * (lambda(0) - lam0_0) / (2.0 * sigma2_lam0) + (lambda(1) - lam1_0) * (lambda(1) - lam1_0) / (2.0 * sigma2_lam1);
  return U(0);
}


void U_lam_grad(arma::vec &U_grad, const arma::vec &lambda, const arma::vec &theta_miss, const arma::vec &theta_obs, 
                const double &lam0_0, const double &lam1_0, const double &sigma2_lam0, const double &sigma2_lam1) {
  arma::vec part1 = normpdf(lambda(0) + lambda(1) * theta_obs) / (1.0 - arma::normcdf(lambda(0) + lambda(1) * theta_obs));
  arma::vec part2 = - normpdf(lambda(0) + lambda(1) * theta_miss) / arma::normcdf(lambda(0) + lambda(1) * theta_miss);
  
  U_grad(0) = arma::sum(part1) + arma::sum(part2) + (lambda(0) - lam0_0) / sigma2_lam0;
  U_grad(1) = arma::sum(part1 % theta_obs) + arma::sum(part2 % theta_miss) + (lambda(1) - lam1_0) / sigma2_lam1;
}


void HMC_lam(arma::vec &lambda_new, double &log_r, const arma::vec &lambda, const arma::vec &theta_miss, const arma::vec &theta_obs, const arma::vec &p_rnorm,
             const double &lam0_0, const double &lam1_0, const double &sigma2_lam0, const double &sigma2_lam1,
             const double &epsilon, const int &num_step) {
  //The kinetic energy has the simplest form arma::sum(p^2)/2
  arma::vec U_grad(2);
  
  lambda_new = lambda;
  arma::vec p_new = p_rnorm;
  
  // a half step for momentum at the beginning
  U_lam_grad(U_grad, lambda_new, theta_miss, theta_obs, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1);
  p_new -= epsilon *  U_grad / 2.0;
  // full steps for position and momentum
  for (int i = 0; i < (num_step-1); i++) {
    lambda_new += epsilon * p_new;
    
    U_lam_grad(U_grad, lambda_new, theta_miss, theta_obs, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1);
    p_new -= epsilon * U_grad;
  }
  lambda_new += epsilon * p_new;
  
  U_lam_grad(U_grad, lambda_new, theta_miss, theta_obs, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1);
  p_new -= epsilon * U_grad / 2.0;
  
  p_new = -p_new;
  log_r = U_lam(lambda, theta_miss, theta_obs, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1) - 
    U_lam(lambda_new, theta_miss, theta_obs, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1) +
    arma::sum(p_rnorm % p_rnorm) / 2.0 - arma::sum(p_new % p_new) / 2.0;
}



void lambda_update(arma::vec &lambda0_t, arma::vec &lambda1_t, const arma::mat &theta_t, const arma::mat &ind_zero, const int &G, 
                   const double &lam0_0, const double &lam1_0, const double &sigma2_lam0, const double &sigma2_lam1,
                   const double &epsilon, const int &num_step) {
  for (int g = 0; g < G; g++) {
    arma::vec lambda_g(2);
    lambda_g(0) = lambda0_t(g);
    lambda_g(1) = lambda1_t(g);
    
    arma::vec theta_g = theta_t.row(g).t();
    arma::uvec ind_0= arma::find(ind_zero.row(g) == true);
    arma::uvec ind_obs = arma::find(ind_zero.row(g) == false);
    
    arma::vec theta_g_obs = theta_g(ind_obs);
    
    arma::vec theta_g_miss = theta_g(ind_0);
    
    arma::vec p_rnorm = arma::randn(2);
    double tmp_unif = arma::randu();
    
    arma::vec lambda_g_new;
    double log_r;
    HMC_lam(lambda_g_new, log_r, lambda_g, theta_g_miss, theta_g_obs, p_rnorm, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1, epsilon, num_step);
    
    if (tmp_unif < exp(log_r) && lambda_g_new(1) < 0) {
      lambda0_t(g) = lambda_g_new(0);
      lambda1_t(g) = lambda_g_new(1);
    }
  }
}


void col_ind_N(arma::mat &Theta_bar, arma::vec &N_k, const arma::mat &theta_t, const arma::vec &C, const int &K, const int &G){
  for(int k = 0; k < K; k++){
    arma::uvec ind  = arma::find(C==k+1);
    N_k(k) = ind.n_elem;
    if(N_k(k) == 0)
      Theta_bar.col(k) = arma::mat(G, 1, fill::zeros);
    else{
      arma::mat tmp = theta_t.cols(ind);
      Theta_bar.col(k) = mean(tmp, 1);
    }
  }
}

void mu_update(arma::mat &mu_t, const arma::mat &theta_t, const arma::mat &p_t, const arma::mat &h_t, const arma::vec &C_t, const arma::vec &sgm_sq_t, 
               const double &sgm_sq_vare_t, const double &eta_mu, const double &tau_mu, 
               const int &G, const int &n, const int &K, const int &S){
  arma::mat ss(K,K); ss.fill(0.0);
  for(int s = 0; s < S; s++)
    ss = ss + p_t.col(s) * p_t.col(s).t();
  arma::mat Sgm_inv_1 = ss/sgm_sq_vare_t;
  arma::vec N_k(K);
  arma::mat Theta_bar(G, K);
  col_ind_N(Theta_bar, N_k, theta_t, C_t, K, G);
  for(int g = 0; g < G; g++){
    arma::vec tmp2 = arma::sum(p_t.each_row() % h_t.row(g), 1) / sgm_sq_vare_t;
    arma::vec sgm_sq_bar = 1/(N_k / sgm_sq_t(g) + 1 / pow(tau_mu, 2));
    arma::vec avg_2 = sgm_sq_bar % (N_k % Theta_bar.row(g).t() / sgm_sq_t(g) + eta_mu/pow(tau_mu, 2));
    arma::mat Sgm_inv_2 = arma::diagmat(1/sgm_sq_bar);
    arma::mat Sgm_inv = Sgm_inv_1 + Sgm_inv_2;
    arma::vec avg_bar = arma::inv_sympd(Sgm_inv) * (tmp2 + Sgm_inv_2 * avg_2);
    mu_t.row(g) =  arma::mvnrnd(avg_bar, arma::inv_sympd(Sgm_inv)).t();
  }
}

void sgm_sq_update(arma::vec &sgm_sq_t, const arma::mat &theta_t, const arma::mat &mu_t, const arma::vec &C_t, 
                   const double &alpha_1, const double &beta_1, 
                   const int &G, const int &n){
  arma::vec s(G); s.fill(beta_1);
  double shape = n/2.0 + alpha_1;
  for(int i = 0; i < n; i++)
    s = s + 0.5 * pow(theta_t.col(i) - mu_t.col(C_t(i)-1), 2);
  arma::vec scale = 1/s;
  for(int g = 0; g < G; g++)
    sgm_sq_t(g) = 1/Rcpp::rgamma(1, shape, scale(g))(0);
}

arma::vec rDir(const arma::vec &alpha){
  int l = alpha.n_elem;
  arma::vec x(l,fill::zeros);
  for(int i = 0; i < l; i++){
    x(i) = Rcpp::rgamma(1, alpha(i), 1.0)(0);
  }
  arma::vec p = x/arma::sum(x);
  return p;
}

void C_update(arma::vec &C_t, const arma::mat &theta_t, const arma::mat &mu_t, const arma::vec &sgm_sq_t,
              const arma::vec &gam_1, const int &n, const int &K){
  arma::vec N_k = zeros<arma::vec>(K);
  arma::uvec fullvec;
  fullvec = regspace<arma::uvec>(1, K);
  for(int k = 0; k < K; k++){
    arma::uvec index = arma::find(C_t == k+1);
    N_k(k) = index.n_elem;
  }
  arma::vec log_pai_i = log(rDir(gam_1 + N_k));
  for(int i = 0 ; i < n; i++){
    arma::vec tmp(K,fill::zeros);
    for(int k =0; k < K; k++){
      arma::vec nor = arma::log_normpdf(theta_t.col(i), mu_t.col(k), sqrt(sgm_sq_t));
      double a = arma::sum(nor);
      tmp(k) = a;
    }
    tmp = tmp + log_pai_i;
    tmp = exp(tmp - max(tmp));
    arma::vec prob = tmp/arma::sum(tmp);
    C_t(i) =  Rcpp::RcppArmadillo::sample(fullvec, 1, false, prob)(0);
  }
}

double logDir(const arma::vec &x, const arma::vec &alpha){
  double dlog = lgamma(arma::sum(alpha)) + arma::sum((alpha - 1) % log(x)) - arma::sum(lgamma(alpha));
  return dlog;
} 

void p_update(arma::mat &p_t, const arma::mat &h_t, const arma::mat &mu_t,  
              const double &sgm_sq_vare_t, const arma::mat &gam_2, 
              const int &S, const int &K, const int &G, const double &gamma_0){
  arma::vec sgm_sq(G); sgm_sq.fill(sgm_sq_vare_t);
  for(int s = 0; s < S; s++){
    arma::vec p_s_star = rDir(gamma_0 * p_t.col(s));
    arma::uvec ind = arma::find(p_s_star == 0);
    if(ind.n_elem != 0)
      continue;
    arma::vec tmp = arma::sum(mu_t.each_row() % p_t.col(s).t(), 1);
    arma::vec tmp_star = arma::sum(mu_t.each_row() % p_s_star.t(), 1);
    arma::vec nor_1 = arma::log_normpdf(h_t.col(s), tmp_star, sqrt(sgm_sq));
    arma::vec nor_2 = arma::log_normpdf(h_t.col(s), tmp, sqrt(sgm_sq));
    double log_ratio = arma::sum(nor_1) - arma::sum(nor_2) + 
      arma::sum((gam_2.col(s) - 1) % log(p_s_star)) - arma::sum((gam_2.col(s)-1) % log(p_t.col(s))) + 
      logDir(p_t.col(s), gamma_0 * p_s_star) - logDir(p_s_star, gamma_0 * p_t.col(s));
    double logu = log(arma::randu<double>());
    if(log_ratio > logu){
      p_t.col(s) = p_s_star;
    }
  }
}


void sgm_sq_vare_update(double &sgm_sq_vare_t, const arma::mat &h_t, const arma::mat &p_t, const arma::mat &mu_t,
                        const double &alpha_vare, const double &beta_vare, 
                        const int &S, const int &G){
  double shape = alpha_vare + 0.5 * S * G;
  double ss = beta_vare;
  for(int s = 0; s < S; s++)
    ss = ss + 0.5 * arma::sum(pow(h_t.col(s) - arma::sum(mu_t.each_row() % p_t.col(s).t(), 1), 2));
  double scale = 1/ss;
  sgm_sq_vare_t = 1/Rcpp::rgamma(1, shape, scale)(0);
}


void h_update(arma::mat &h, const arma::mat &Z, const arma::mat &p_t, const arma::mat &mu_t,
              const arma::vec &sgm_sq_star_t, const arma::vec &R_t,
              const double &sgm_sq_vare_t, const int &S, const int &G){
  arma::vec Z_bar(G);
  for(int s = 0; s < S; s++){
    arma::uvec index = arma::find(R_t == s+1);
    int N_s = index.n_elem;
    if(N_s == 0)
      Z_bar = arma::vec(G, fill::zeros);
    else
      Z_bar = arma::sum(Z.cols(index), 1);
    arma::vec tmp = arma::sum(mu_t.each_row() % p_t.col(s).t(), 1);
    arma::vec sgm_sq_hat, h_hat;
    sgm_sq_hat = 1/(N_s / sgm_sq_star_t  + 1 / sgm_sq_vare_t);
    h_hat = (Z_bar / sgm_sq_star_t + tmp / sgm_sq_vare_t) % sgm_sq_hat;
    h.col(s) = h_hat + arma::randn<arma::vec>(G) % sqrt(sgm_sq_hat);
  }
}


void sgm_sq_star_update(arma::vec &sgm_sq_star_t, const arma::mat &Z, const arma::vec &R_t, const arma::mat &h_t,
                        const int &S, const int &G, const int &m,
                        const double &alpha_2, const double &beta_2){
  
  arma::vec s(G); s.fill(beta_2);
  double shape = m/2.0 + alpha_2;
  for(int i = 0; i < m; i++){
    s = s + 0.5 * pow(Z.col(i) - h_t.col(R_t(i)-1), 2);
  }
  arma::vec scale = 1/s;
  for(int g = 0; g < G; g++){
    sgm_sq_star_t(g) = 1/Rcpp::rgamma(1, shape, scale(g))(0);
  }
}

arma::vec neigh_index(int i, arma::mat ind_mat){
  //coordinate of index i-th row (i = 0,..., m)
  int l = ind_mat(i,0);
  int w = ind_mat(i,1);
  arma::vec tmp;
  tmp << l << l-1 << l+1 << l <<  w-1<< w << w << w+1;
  arma::mat Tmp = arma::mat(tmp);
  Tmp.set_size(4,2);
  // Tmp is the neighbor coordinates of (l,w)
  arma::vec nei_ind(4); nei_ind.fill(0.1);
  for(int j = 0; j < 4; j++){
    for(unsigned int nr = 0; nr < ind_mat.n_rows; nr++){
      arma::vec tmp_1 = (ind_mat.row(nr) - Tmp.row(j)).t();
      double a = tmp_1.is_zero();
      if(a > 0){
        nei_ind(j) = nr;
        break;
      }
    }
  }
  arma::uvec nei = arma::find(nei_ind != 0.1);
  arma::vec nei_index = nei_ind(nei);
  return nei_index;
}

//update the region type *vector*.
arma::vec R_update(const arma::mat &Z, const arma::mat &h_t, const arma::mat &psi_t, 
                   const arma::mat &ind_mat, arma::vec R_t, const arma::vec &sgm_sq_star_t, 
                   const int &S, const int &G, const int &m){
  arma::uvec fullvec;
  fullvec = regspace<arma::uvec>(1, S);
  for(int i = 0; i < m; i++){
    arma::vec neigh = neigh_index(i, ind_mat);
    arma::vec tmp(S,fill::zeros);
    for(int s = 0; s < S; s++){
      arma::vec nor = arma::log_normpdf(Z.col(i), h_t.col(s), sqrt(sgm_sq_star_t));
      double a = arma::sum(nor);
      for(unsigned int j = 0; j < neigh.n_elem; j++){
        if(R_t(neigh(j)) != s+1){
          a += psi_t(s,R_t(neigh(j)) - 1);
        }
      }
      tmp(s) = a;
    }
    tmp = exp(tmp - tmp.max());
    arma::vec prob = tmp/arma::sum(tmp);
    R_t(i) =  Rcpp::RcppArmadillo::sample(fullvec, 1, false, prob)(0);
  }
  return R_t;
}


//calculate H(R|psi) function.
double H_fun(const arma::vec &R_t, const arma::mat &psi_t, const arma::mat &ind_mat, const int &m){
  double sum_H = 0.0;
  for(int i = 0; i < m; i++){
    arma::vec nei = neigh_index(i, ind_mat);
    for(unsigned int j = 0; j < nei.n_elem; j++){
      if(R_t(nei(j)) != R_t(i)){
        sum_H += psi_t(R_t(i)-1, R_t(nei(j))-1);
      }
    }
  }
  return - sum_H / 2.0;
}


void psi_update(arma::mat &psi_t, const arma::mat &Z, const arma::mat &h_t, const arma::mat &ind_mat,
                const arma::vec &sgm_sq_star_t,  const arma::vec &R_t,
                const int &S, const int &G, const int &m, 
                const double &tau_0, const double &eta_psi, const double &tau_psi){
  //sample psi matrix from proposal distribution (normal)
  //upper triangular matrix.
  for(int i = 1; i < S; i++){
    for(int j = 0; j < i; j++){
      double psi_star_ji = arma::randn<double>() * tau_0 + psi_t(j,i);
      arma::mat psi_star = psi_t;
      psi_star(j, i) = psi_star_ji;
      psi_star(i, j) = psi_star(j, i);
      arma::vec y = R_update(Z, h_t, psi_star, ind_mat, R_t, sgm_sq_star_t,S,G,m);
      double log_ratio = arma::log_normpdf(psi_star_ji, eta_psi, tau_psi) -
        arma::log_normpdf(psi_t(i, j), eta_psi, tau_psi) +
        H_fun(R_t, psi_t, ind_mat, m) + H_fun(y, psi_star, ind_mat, m) -
        H_fun(R_t, psi_star, ind_mat, m) - H_fun(y, psi_t, ind_mat, m);
      
      double logu = log(arma::randu<double>());
      if(log_ratio > logu){
        psi_t(j, i) = psi_star_ji;
        psi_t(i, j) = psi_star_ji;
      }
    }
  }
}

// [[Rcpp::export]]
List MCMC_full(const int n_iter, const int n_save, arma::mat Z, arma::mat theta_t, arma::mat ind_zero,
               arma::mat mu_t, arma::mat p_t, arma::mat h_t, arma::mat psi_t, arma::mat ind_mat, 
               arma::vec lambda0_t, arma::vec lambda1_t, arma::vec C_t, arma::vec R_t, 
               arma::vec sgm_sq_t, arma::vec sgm_sq_star_t, arma::vec gam_1, arma::mat gam_2, 
               int G, int n, int m, int K, int S,
               double sgm_sq_vare_t, double eta_mu = 0, double tau_mu = 5, 
               double alpha_1 = 2, double beta_1 = 0.1, double gamma_0 = 1000.0,
               double alpha_vare = 3, double beta_vare = 0.1, 
               double alpha_2 = 5, double beta_2 = 0.01,
               double tau_0 = 0.1, double eta_psi = 0, double tau_psi = 10,
               double epsilon_theta = 0.2, int num_step_theta = 20,
               double lam0_0 = 2, double lam1_0 = -2, double sigma2_lam0 = 0.25, 
               double sigma2_lam1 = 0.25, double epsilon_lam = 0.01, int num_step_lam = 10,
               bool iter_save = false, int iter_print = 1000, bool class_print = false) {
  
  int save_start = n_iter - n_save;
  arma::mat C_save(K, n, fill::zeros);
  arma::mat R_save(S, m, fill::zeros);
  arma::mat mu_save(G, K, fill::zeros);
  arma::mat p_save(K, S, fill::zeros);
  arma::mat H(G, S, fill::zeros);
  arma::vec lam0_save(G, fill::zeros);
  arma::vec lam1_save(G, fill::zeros);
  arma::mat psi_save(S, S, fill::zeros);
  
  if (iter_save) {
    arma::mat C_iter(n, n_save);
    arma::mat R_iter(m, n_save);
    arma::cube mu_iter(G, K, n_save);
    arma::mat sgm_sq_iter(G, n_save);
    arma::cube h_iter(G, S, n_save);
    arma::vec sgm_sq_vare_iter(n_save);
    arma::mat sgm_sq_star_iter(G, n_save);
    arma::cube psi_iter(S, S, n_save);
    arma::cube p_iter(K, S, n_save);
    
    //////save the results in the iterations for the first column of each matrix
    arma::mat theta_iter(G, n_save);
    ///////////////////////////////
    
    arma::mat lam0_iter(G, n_save);
    arma::mat lam1_iter(G, n_save);
    
    if (class_print) {
      Environment base("package:base"); 
      Function table = base["table"];
      
      SEXP Cell_table = table(Rcpp::_["..."] = C_t);
      irowvec Cell_out = Rcpp::as<arma::irowvec>(Cell_table);
      SEXP Region_table = table(Rcpp::_["..."] = R_t);
      irowvec Region_out = Rcpp::as<arma::irowvec>(Region_table);
      cout<<"Iteration "<< 0 <<endl;
      cout<<"Cell"<<endl; 
      cout<<Cell_out<<endl; 
      cout<<"Region"<<endl; 
      cout<<Region_out<<endl; 
      
      for (int t_iter = 0; t_iter < n_iter; t_iter++) {
        theta_update(theta_t, ind_zero, mu_t, sgm_sq_t, lambda0_t, lambda1_t, C_t, n, epsilon_theta, num_step_theta);
        
        lambda_update(lambda0_t, lambda1_t, theta_t, ind_zero, G, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1, epsilon_lam, num_step_lam);
        
        mu_update(mu_t, theta_t, p_t, h_t, C_t, sgm_sq_t, sgm_sq_vare_t, eta_mu, tau_mu, G, n, K, S);
        
        C_update(C_t, theta_t, mu_t, sgm_sq_t, gam_1, n, K);
        
        sgm_sq_update(sgm_sq_t, theta_t, mu_t, C_t, alpha_1, beta_1, G, n);
        
        p_update(p_t, h_t, mu_t, sgm_sq_vare_t, gam_2, S, K, G, gamma_0);
        
        sgm_sq_vare_update(sgm_sq_vare_t, h_t, p_t, mu_t, alpha_vare, beta_vare, S, G);
        
        h_update(h_t, Z, p_t, mu_t, sgm_sq_star_t, R_t, sgm_sq_vare_t, S, G);
        
        sgm_sq_star_update(sgm_sq_star_t, Z, R_t, h_t, S, G, m, alpha_2, beta_2);  
        
        R_t = R_update(Z, h_t, psi_t, ind_mat, R_t, sgm_sq_star_t, S, G, m);
        
        psi_update(psi_t, Z, h_t, ind_mat, sgm_sq_star_t, R_t, S, G, m, tau_0, eta_psi, tau_psi);
        
        SEXP Cell_table = table(Rcpp::_["..."] = C_t);
        irowvec Cell_out = Rcpp::as<arma::irowvec>(Cell_table);
        SEXP Region_table = table(Rcpp::_["..."] = R_t);
        irowvec Region_out = Rcpp::as<arma::irowvec>(Region_table);
        cout<<"Iteration "<< t_iter+1 <<endl;
        cout<<"Cell"<<endl; 
        cout<<Cell_out<<endl; 
        cout<<"Region"<<endl; 
        cout<<Region_out<<endl; 
        
        if (t_iter >= save_start) {
          int save_i = t_iter - save_start;
          for (int C_i = 0; C_i < n; C_i++) {
            C_save((C_t(C_i)-1),C_i) += 1;
          }
          for (int R_i = 0; R_i < m; R_i++) {
            R_save((R_t(R_i)-1),R_i) += 1;
          }
          
          mu_save += mu_t;
          p_save += p_t;
          H += h_t;
          lam0_save += lambda0_t;
          lam1_save += lambda1_t;
          psi_save += psi_t;
          
          C_iter.col(save_i) = C_t;
          R_iter.col(save_i) = R_t;
          mu_iter.slice(save_i) = mu_t;
          p_iter.slice(save_i) = p_t;
          h_iter.slice(save_i) = h_t;
          
          sgm_sq_iter.col(save_i) = sgm_sq_t;
          sgm_sq_vare_iter(save_i) = sgm_sq_vare_t;
          sgm_sq_star_iter.col(save_i) = sgm_sq_star_t;
          psi_iter.slice(save_i) = psi_t;
          theta_iter.col(save_i) = theta_t.col(0);
          
          lam0_iter.col(save_i) = lambda0_t;
          lam1_iter.col(save_i) = lambda1_t;
        }
      }
      
      mu_save /= n_save;
      p_save /= n_save;
      H /= n_save;
      lam0_save /= n_save;
      lam1_save /= n_save;
      psi_save /= n_save;
      
      arma::vec C_est(n);
      for (int C_i = 0; C_i < n; C_i++) {
        C_est(C_i) = C_save.col(C_i).index_max() + 1;
      }
      arma::vec R_est(m);
      for (int R_i = 0; R_i < m; R_i++) {
        R_est(R_i) = R_save.col(R_i).index_max() + 1;
      }
      
      return List::create(Named("cell_labels")=C_est, Named("region_labels")=R_est, Named("cell_type_mean_expr")=mu_save, Named("prop")=p_save, Named("region_mean_expr")=H,
                          Named("lam0")=lam0_save, Named("lam1")=lam1_save, Named("region_interaction")=psi_save,
                                Named("cell_label_post")=C_iter, Named("region_label_post")=R_iter, Named("region_interaction_post")=psi_iter, Named("prop_post")=p_iter,
                                      Named("cell_type_mean_expr_post")=mu_iter,  Named("lam0_post")=lam0_iter, Named("lam1_post")=lam1_iter,
                                      Named("sgm_sq_y_post")=sgm_sq_iter, Named("sgm_sq_h_post")=sgm_sq_vare_iter, 
                                      Named("sgm_sq_z_post")=sgm_sq_star_iter, Named("theta_post")=theta_iter, Named("region_mean_expr_post") = h_iter);
    } else {
      for (int t_iter = 0; t_iter < n_iter; t_iter++) {
        theta_update(theta_t, ind_zero, mu_t, sgm_sq_t, lambda0_t, lambda1_t, C_t, n, epsilon_theta, num_step_theta);
        
        lambda_update(lambda0_t, lambda1_t, theta_t, ind_zero, G, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1, epsilon_lam, num_step_lam);
        
        mu_update(mu_t, theta_t, p_t, h_t, C_t, sgm_sq_t, sgm_sq_vare_t, eta_mu, tau_mu, G, n, K, S);
        
        C_update(C_t, theta_t, mu_t, sgm_sq_t, gam_1, n, K);
        
        sgm_sq_update(sgm_sq_t, theta_t, mu_t, C_t, alpha_1, beta_1, G, n);
        
        p_update(p_t, h_t, mu_t, sgm_sq_vare_t, gam_2, S, K, G, gamma_0);
        
        sgm_sq_vare_update(sgm_sq_vare_t, h_t, p_t, mu_t, alpha_vare, beta_vare, S, G);
        
        h_update(h_t, Z, p_t, mu_t, sgm_sq_star_t, R_t, sgm_sq_vare_t, S, G);
        
        sgm_sq_star_update(sgm_sq_star_t, Z, R_t, h_t, S, G, m, alpha_2, beta_2);  
        
        R_t = R_update(Z, h_t, psi_t, ind_mat, R_t, sgm_sq_star_t, S, G, m);
        
        psi_update(psi_t, Z, h_t, ind_mat, sgm_sq_star_t, R_t, S, G, m, tau_0, eta_psi, tau_psi);
        
        if ((t_iter+1) % iter_print == 0) {
          cout<<"Iteration "<< t_iter+1 <<endl;
        }
        
        if (t_iter >= save_start) {
          int save_i = t_iter - save_start;
          for (int C_i = 0; C_i < n; C_i++) {
            C_save((C_t(C_i)-1),C_i) += 1;
          }
          for (int R_i = 0; R_i < m; R_i++) {
            R_save((R_t(R_i)-1),R_i) += 1;
          }
          
          mu_save += mu_t;
          p_save += p_t;
          H += h_t;
          lam0_save += lambda0_t;
          lam1_save += lambda1_t;
          psi_save += psi_t;
          
          C_iter.col(save_i) = C_t;
          R_iter.col(save_i) = R_t;
          mu_iter.slice(save_i) = mu_t;
          p_iter.slice(save_i) = p_t;
          h_iter.slice(save_i) = h_t;
          
          sgm_sq_iter.col(save_i) = sgm_sq_t;
          sgm_sq_vare_iter(save_i) = sgm_sq_vare_t;
          sgm_sq_star_iter.col(save_i) = sgm_sq_star_t;
          psi_iter.slice(save_i) = psi_t;
          theta_iter.col(save_i) = theta_t.col(0);
          
          lam0_iter.col(save_i) = lambda0_t;
          lam1_iter.col(save_i) = lambda1_t;
        }
      }
      
      mu_save /= n_save;
      p_save /= n_save;
      H /= n_save;
      lam0_save /= n_save;
      lam1_save /= n_save;
      psi_save /= n_save;
      
      arma::vec C_est(n);
      for (int C_i = 0; C_i < n; C_i++) {
        C_est(C_i) = C_save.col(C_i).index_max() + 1;
      }
      arma::vec R_est(m);
      for (int R_i = 0; R_i < m; R_i++) {
        R_est(R_i) = R_save.col(R_i).index_max() + 1;
      }
      
      return List::create(Named("cell_labels")=C_est, Named("region_labels")=R_est, Named("cell_type_mean_expr")=mu_save, Named("prop")=p_save, Named("region_mean_expr")=H,
                          Named("lam0")=lam0_save, Named("lam1")=lam1_save, Named("region_interaction")=psi_save,
                                Named("cell_label_post")=C_iter, Named("region_label_post")=R_iter, Named("region_interaction_post")=psi_iter, Named("prop_post")=p_iter,
                                      Named("cell_type_mean_expr_post")=mu_iter,  Named("lam0_post")=lam0_iter, Named("lam1_post")=lam1_iter,
                                      Named("sgm_sq_y_post")=sgm_sq_iter, Named("sgm_sq_h_post")=sgm_sq_vare_iter, 
                                      Named("sgm_sq_z_post")=sgm_sq_star_iter, Named("theta_post")=theta_iter, Named("region_mean_expr_post") = h_iter);
    }
  } else {
    
    if (class_print) {
      Environment base("package:base"); 
      Function table = base["table"];
      
      SEXP Cell_table = table(Rcpp::_["..."] = C_t);
      irowvec Cell_out = Rcpp::as<arma::irowvec>(Cell_table);
      SEXP Region_table = table(Rcpp::_["..."] = R_t);
      irowvec Region_out = Rcpp::as<arma::irowvec>(Region_table);
      cout<<"Iteration "<< 0 <<endl;
      cout<<"Cell"<<endl; 
      cout<<Cell_out<<endl; 
      cout<<"Region"<<endl; 
      cout<<Region_out<<endl; 
      
      for (int t_iter = 0; t_iter < n_iter; t_iter++) {
        theta_update(theta_t, ind_zero, mu_t, sgm_sq_t, lambda0_t, lambda1_t, C_t, n, epsilon_theta, num_step_theta);
        
        lambda_update(lambda0_t, lambda1_t, theta_t, ind_zero, G, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1, epsilon_lam, num_step_lam);
        
        mu_update(mu_t, theta_t, p_t, h_t, C_t, sgm_sq_t, sgm_sq_vare_t, eta_mu, tau_mu, G, n, K, S);
        
        C_update(C_t, theta_t, mu_t, sgm_sq_t, gam_1, n, K);
        
        sgm_sq_update(sgm_sq_t, theta_t, mu_t, C_t, alpha_1, beta_1, G, n);
        
        p_update(p_t, h_t, mu_t, sgm_sq_vare_t, gam_2, S, K, G, gamma_0);
        
        sgm_sq_vare_update(sgm_sq_vare_t, h_t, p_t, mu_t, alpha_vare, beta_vare, S, G);
        
        h_update(h_t, Z, p_t, mu_t, sgm_sq_star_t, R_t, sgm_sq_vare_t, S, G);
        
        sgm_sq_star_update(sgm_sq_star_t, Z, R_t, h_t, S, G, m, alpha_2, beta_2);  
        
        R_t = R_update(Z, h_t, psi_t, ind_mat, R_t, sgm_sq_star_t, S, G, m);
        
        psi_update(psi_t, Z, h_t, ind_mat, sgm_sq_star_t, R_t, S, G, m, tau_0, eta_psi, tau_psi);
        
        SEXP Cell_table = table(Rcpp::_["..."] = C_t);
        irowvec Cell_out = Rcpp::as<arma::irowvec>(Cell_table);
        SEXP Region_table = table(Rcpp::_["..."] = R_t);
        irowvec Region_out = Rcpp::as<arma::irowvec>(Region_table);
        cout<<"Iteration "<< t_iter+1 <<endl;
        cout<<"Cell"<<endl; 
        cout<<Cell_out<<endl; 
        cout<<"Region"<<endl; 
        cout<<Region_out<<endl; 
        
        if (t_iter >= save_start) {
          for (int C_i = 0; C_i < n; C_i++) {
            C_save((C_t(C_i)-1),C_i) += 1;
          }
          for (int R_i = 0; R_i < m; R_i++) {
            R_save((R_t(R_i)-1),R_i) += 1;
          }
          
          mu_save += mu_t;
          p_save += p_t;
          H += h_t;
          lam0_save += lambda0_t;
          lam1_save += lambda1_t;
          psi_save += psi_t;
        }
      }
      
      mu_save /= n_save;
      p_save /= n_save;
      H /= n_save;
      lam0_save /= n_save;
      lam1_save /= n_save;
      psi_save /= n_save;
      
      arma::vec C_est(n);
      for (int C_i = 0; C_i < n; C_i++) {
        C_est(C_i) = C_save.col(C_i).index_max() + 1;
      }
      arma::vec R_est(m);
      for (int R_i = 0; R_i < m; R_i++) {
        R_est(R_i) = R_save.col(R_i).index_max() + 1;
      }
      
      return List::create(Named("cell_labels")=C_est, Named("region_labels")=R_est, Named("cell_type_mean_expr")=mu_save, Named("prop")=p_save,
                                Named("region_mean_expr")=H, Named("lam0")=lam0_save, Named("lam1")=lam1_save, Named("region_interaction")=psi_save);
    } else {
      
      for (int t_iter = 0; t_iter < n_iter; t_iter++) {
        theta_update(theta_t, ind_zero, mu_t, sgm_sq_t, lambda0_t, lambda1_t, C_t, n, epsilon_theta, num_step_theta);
        
        lambda_update(lambda0_t, lambda1_t, theta_t, ind_zero, G, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1, epsilon_lam, num_step_lam);
        
        mu_update(mu_t, theta_t, p_t, h_t, C_t, sgm_sq_t, sgm_sq_vare_t, eta_mu, tau_mu, G, n, K, S);
        
        C_update(C_t, theta_t, mu_t, sgm_sq_t, gam_1, n, K);
        
        sgm_sq_update(sgm_sq_t, theta_t, mu_t, C_t, alpha_1, beta_1, G, n);
        
        p_update(p_t, h_t, mu_t, sgm_sq_vare_t, gam_2, S, K, G, gamma_0);
        
        sgm_sq_vare_update(sgm_sq_vare_t, h_t, p_t, mu_t, alpha_vare, beta_vare, S, G);
        
        h_update(h_t, Z, p_t, mu_t, sgm_sq_star_t, R_t, sgm_sq_vare_t, S, G);
        
        sgm_sq_star_update(sgm_sq_star_t, Z, R_t, h_t, S, G, m, alpha_2, beta_2);  
        
        R_t = R_update(Z, h_t, psi_t, ind_mat, R_t, sgm_sq_star_t, S, G, m);
        
        psi_update(psi_t, Z, h_t, ind_mat, sgm_sq_star_t, R_t, S, G, m, tau_0, eta_psi, tau_psi);
        
        if ((t_iter+1) % iter_print == 0) {
          cout<<"Iteration "<< t_iter+1 <<endl;
        }
        
        if (t_iter >= save_start) {
          for (int C_i = 0; C_i < n; C_i++) {
            C_save((C_t(C_i)-1),C_i) += 1;
          }
          for (int R_i = 0; R_i < m; R_i++) {
            R_save((R_t(R_i)-1),R_i) += 1;
          }
          
          mu_save += mu_t;
          p_save += p_t;
          H += h_t;
          lam0_save += lambda0_t;
          lam1_save += lambda1_t;
          psi_save += psi_t;
        }
      }
      
      mu_save /= n_save;
      p_save /= n_save;
      H /= n_save;
      lam0_save /= n_save;
      lam1_save /= n_save;
      psi_save /= n_save;
      
      arma::vec C_est(n);
      for (int C_i = 0; C_i < n; C_i++) {
        C_est(C_i) = C_save.col(C_i).index_max() + 1;
      }
      arma::vec R_est(m);
      for (int R_i = 0; R_i < m; R_i++) {
        R_est(R_i) = R_save.col(R_i).index_max() + 1;
      }
      
      return List::create(Named("cell_labels")=C_est, Named("region_labels")=R_est, Named("cell_type_mean_expr")=mu_save, Named("prop")=p_save,
                                Named("region_mean_expr")=H, Named("lam0")=lam0_save, Named("lam1")=lam1_save, Named("region_interaction")=psi_save);
    }
    
  }
}
