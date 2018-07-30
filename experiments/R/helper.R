evaluate_link_prediction = function(true, predicted){
  diag(true) = NA
  diag(predicted) = NA
  tp = sum(true * predicted, na.rm = TRUE)
  fp = sum((true == 0) * predicted, na.rm = TRUE)
  P = sum(true, na.rm = TRUE)
  N = sum(true == 0, na.rm = TRUE)
  tpr = tp / P
  fpr = fp / N
  recall = tp/P
  precision = tp / (tp + fp)
  f = 2. / (1./ precision + 1./ recall)
  data.frame(tp, fp, P, N, tpr, fpr, recall, precision, f)
}

# finds from row to col
find_xcorr_matrix = function(X, max_lag){
  N = ncol(X)
  output = matrix(ncol = N, nrow = N)
  for (i in 1:N){
    for (j in 1:N){
      if (i != j){
        output[i,j] = find_xcorr(X[,i], X[,j], max_lag)
      }
    }
  }
  diag(output) = NA
  output
}

# from Y to X
find_xcorr = function(Y, X, max_lag){
  aa = ccf(X, Y, lag.max = max_lag, plot = F, type = c("correlation"))
  ibegin = match(0, aa$lag)
  iend = match(max_lag, aa$lag)
  sum(aa$acf[ibegin:iend])
}

library(zoo)

get_AUC = function(suffStat, func, data, true_labels){
  pc.fit = func(suffStat, indepTest = gaussCItest, p = ncol(data), alpha = 0.8)
  all_p = unique(sort(as.vector(pc.fit@pMax)))
  p_levels = all_p
  if (length(all_p) > 1){
    p_levels = unique(sort(as.vector(pc.fit@pMax) )) + min(diff(all_p)) / 2
  }
  index_p = 1
  output = list()
  for (alpha in p_levels){
    pc.fit = func(suffStat, indepTest = gaussCItest, p = ncol(data), alpha = alpha)
    pred_net = as(pc.fit@graph, "matrix")
    pred = pred_net[upper.tri(pred_net) | lower.tri(pred_net)]
    output[[index_p]] = extract_measures(true_labels, pred)
    index_p = index_p + 1
  }
  AUC = extract_AUC(as.data.frame(rbindlist(output)))  
  AUC
}

get_PC_AUC = function(suffStat, conservative, data, true_labels){
  pc.fit = pc(suffStat, indepTest = gaussCItest, p = ncol(data), alpha = 0.8, conservative = conservative, u2pd = "relaxed")
  all_p = unique(sort(as.vector(pc.fit@pMax)))
  p_levels = all_p
  if (length(all_p) > 1){
    p_levels = unique(sort(as.vector(pc.fit@pMax) )) + min(diff(all_p)) / 2
  }
  index_p = 1
  output = list()
  for (alpha in p_levels){
    pc.fit = pc(suffStat, indepTest = gaussCItest, p = ncol(data), alpha = alpha, conservative = conservative, u2pd = "relaxed")
    pred_net = as(pc.fit@graph, "matrix")
    pred = pred_net[upper.tri(pred_net) | lower.tri(pred_net)]
    output[[index_p]] = extract_measures(true_labels, pred)
    index_p = index_p + 1
  }
  AUC = extract_AUC(as.data.frame(rbindlist(output)))  
  AUC
}


require(ppcor)
get_AUC_pwling = function(data, true_labels){
  est = pcor(data, method = "kendall")$estimate
  direc = pwling(t(as.matrix(data)))
  pred_net = abs(est) * ((direc > 0) * 1)
  pred = pred_net[upper.tri(pred_net) | lower.tri(pred_net)]
  perf_pred = prediction(pred, true_labels)
  performance(perf_pred, "auc")@y.values[[1]]  
}


get_AUC_bnlearn = function(data, true_labels, func){
  library(bnlearn) 
  
  corrs = rcorr(as.matrix(data), type="pearson")

  all_p = unique(sort(as.vector(corrs$P)))
  p_levels = all_p
  if (length(all_p) > 1){
    p_levels = unique(sort(as.vector(corrs$P))) + min(diff(all_p)) / 2
  }
  p_levels[p_levels > 1] = 1
  index_p = 1
  output = list()
  for (alpha in p_levels){
    bn_fit = func(data, alpha =  alpha)
    pred_net = amat(bn_fit)
    diag(pred_net) = 0
#     pred_net = adj.remove.cycles(adjmat = 
#                                    matrix(pred_net, nrow = nrow(pred_net), ncol = ncol(pred_net)) * 1, maxlength = 10)$adjmat.acyclic
    pred = pred_net[upper.tri(pred_net) | lower.tri(pred_net)]
    output[[index_p]] = data.frame(extract_measures(true_labels, pred), threshod = alpha)
    index_p = index_p + 1
  }
  AUC = extract_AUC(as.data.frame(rbindlist(output)))  
  AUC
}


# library(predictionet)
# 
# get_AUC_p_DAG = function(p, true_labels){
#   all_p = unique(sort(p))
#   p_levels = all_p
#   if (length(all_p) > 1){
#     p_levels = unique(sort(as.vector(p) )) - min(diff(all_p)) / 2
#   }
#   index_p = 1
#   output = list()
#   for (alpha in p_levels){
#     pred_net = p > alpha
#     diag(pred_net) = 0
# #      pred_net = adj.remove.cycles(adjmat = 
# #                   matrix(pred_net, nrow = nrow(pred_net), ncol = ncol(pred_net)) * 1, maxlength = 10)$adjmat.acyclic
#     pred = pred_net[upper.tri(pred_net) | lower.tri(pred_net)]
#     output[[index_p]] = data.frame(extract_measures(true_labels, pred), thresh = alpha)
#     index_p = index_p + 1
#   }
#   AUC = extract_AUC(as.data.frame(rbindlist(output)))  
#   AUC
# }

require(zoo)
extract_AUC = function(output){
  output = output[,c("tpr", "fpr")]
  output = rbind(data.frame(tpr = 0, fpr = 0), output)
  output = rbind(data.frame(tpr = 1, fpr = 1), output)
  output = output[order(output$tpr),]
  output = output[order(output$fpr),]
  
  ########### comment below for getting uopper bound on AUC ###################
  #max_i = output[1,]$tpr
  #rem = c(FALSE)
  #for(i in 2:nrow(output)){
  #  if (output[i,]$tpr < max_i) {
  #    rem = c(rem, TRUE)
  #  }else{
  #    max_i = output[i,]$tpr
  #    rem = c(rem, FALSE)
  #  }
  #}
  #output = output[!rem,]
  ############## until here ####################################
  output = rbind(data.frame(tpr = 0, fpr = 0), output)
  output = rbind(data.frame(tpr = 1, fpr = 1), output)
  output = output[order(output$tpr),]
  output = output[order(output$fpr),]
  id <- order(output$fpr)
  AUC <- sum(diff(output$fpr[id])*rollmean(output$tpr[id],2))
  AUC
}

extract_measures = function(true, predicted){
  tp = sum(true * predicted, na.rm = TRUE)
  fp = sum((true == 0) * predicted, na.rm = TRUE)
  P = sum(true, na.rm = TRUE)
  N = sum(true == 0, na.rm = TRUE)
  tpr = tp / P
  fpr = fp / N
  recall = tp/P
  precision = tp / (tp + fp)
  f = 2. / (1./ precision + 1./ recall)
  data.frame(tp, fp, P, N, tpr, fpr, recall, precision, f)
}