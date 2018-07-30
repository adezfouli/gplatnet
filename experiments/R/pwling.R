#PWLING: pairwise causality measures in linear non-Gaussian model
#Version 1.2, Aapo Hyvarinen, Feb 2013
#Input: Data matrix with variables as rows, and index of method [1...5]
#Output: Matrix LR with likelihood ratios
#        If entry (i,j) in that matrix is positive,
#        estimate of causal direction is i -> j
#Methods 1...5 are as follows:
#  1: General entropy-based method, for variables of any distribution
#  2: First-order approximation of LR by tanh, for sparse variables
#  3: Basic skewness measure, for skewed variables
#  4: New skewness-based measure, robust to outliers
#  5: Dodge-Rousson measure, for skewed variables
#  If you want to use method 3 or 4 without skewness correction,
#     input -3 or -4 as the method.
#See http://www.cs.helsinki.fi/u/ahyvarin/code/pwcausal/ for more information

library(matrixStats)

pwling = function(X){


#Get size parameters
n = dim(X)[1]
Ti = dim(X)[2]

eps = 2.2204e-16
#Standardize each variable
X <- X-rowMeans(X)
X <- X/((rowSds(X)+eps))


#Compute covariance matrix
    C <- cov(t(X))


#Compute causality measures
###########################

    #Initialize output matrix
    LR <- matrix(0, nrow = n, ncol = n)
    #Loop throgh pairs
    for (i in 1:n){
      for (j in 1:n){
        if (i!=j){
          res1 <- (X[j,]-C[j,i]*X[i,])
          res2 <- (X[i,]-C[i,j]*X[j,])
            LR[i,j] <- (mentappr(X[j,])-mentappr(X[i,])
                     -mentappr(res1)+mentappr(res2))[1,1]
        }
      }
    }
    LR
}


#MENTAPPR: Compute MaxEnt approximation of negentropy and differential entropy
#Aapo Hyvarinen, May 2012
#Based on NIPS*97 paper, www.cs.helsinki.fi/u/ahyvarin/papers/NIPS97.pdf
#  but using a new nonlinearity
#Input: sample of continous-valued random variable as a vector.
#        For matrices, entropy computed for each column
#Output: (differential) entropy and, optionally, negentropy

mentappr = function(x){
  
  #standardize
  x <- x-mean(x)
  xstd <- sd(x)
  x <- x/xstd
  
  #Constants we need
  k1 <- 36/(8*sqrt(3)-9)
  gamma <- 0.37457
  k2 <- 79.047
  gaussianEntropy <- log(2*pi)/2+1/2
  
  #This is negentropy
  negentropy <- k2*(mean(log(cosh(x)))-gamma)^2+k1*mean(x*exp(-x^2/2))^2
  
  #This is entropy
  entropy <- gaussianEntropy - negentropy + log(xstd)
  data.frame(en = entropy, nen = negentropy)
}
