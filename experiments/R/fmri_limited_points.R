# This file calculates AUC for Latnet and baseline methods reported in the paper

library(bnlearn)
library(data.table)
library(pcalg)
library(ROCR)
library(Hmisc)
library(ggplot2)
library(gridExtra)
library(extrafont)

pdf_output_path = '../../nongit/fmri/'
results_output_path = '../../nongit/fmri/'
fmri_data_path = '../data/fmri_sim/'
latnet_results_path = '../../nongit/archive/fmri/'

source('helper.R')
source('pwling.R')

# which datasets to use for running experiments - these refer to datasets in Smiths et al 2011 Neuroimaging
exprlist = c(1, 2, 3)

# name of methods
methods_list = c("LATNET", "PC", "CPC", "LiNGAM", "IAMB", "PW-LiNGAM")

all_results = NA

# these are not being used
expr = 1
subject = 4
Ti = 50

index_r = 1
results_list = list()
for (expr in exprlist) {
    for (subject in 1 : 50) {

        # how many datapoints to include
        for (Ti in c(50, 100, 200)) {
            print(sprintf('subject %d, expr %d, T %d', subject, expr, Ti))

            # path to file containing time-series
            input_file = paste(fmri_data_path, 'ts_sim', expr, '.csv', sep = "")

            # path to file containing underlying network
            conn_file = paste(fmri_data_path, 'net_sim', expr, '.csv', sep = "")

            all_fmri_data = read.csv(input_file, header = F)
            time_points = nrow(all_fmri_data) / 50
            all_true_conn = as.matrix(read.csv(conn_file, header = F))
            fmri_data = all_fmri_data[((subject - 1) * time_points + 1) : (subject * time_points),][1 : Ti,]

            # shuffling order of of nodes
            row_index = sample(ncol(fmri_data))
            fmri_data = fmri_data[, row_index]
            true_conn = matrix(all_true_conn[subject,], nrow = ncol(fmri_data))
            true_conn = true_conn[row_index, row_index]
            diag(true_conn) = NA
            true_labels = true_conn != 0
            true_labels = true_labels[upper.tri(true_labels) | lower.tri(true_labels)]

            if ("LATNET" %in% methods_list) {
                method = "LATNET"
                latnet_add = paste(latnet_results_path, 'fmri_sim', expr, '_LatNet/', Ti, '/latnet_', (subject - 1), '/p.csv', sep = "")
                latnet_res = read.csv(latnet_add, header = F)
                pred_net = latnet_res
                pred_net = t(abs(pred_net))
                pred_net = pred_net[row_index, row_index]

                pred = pred_net[upper.tri(pred_net) | lower.tri(pred_net)]
                perf_pred = prediction(pred, true_labels * 1)
                perf = performance(perf_pred, c("auc"))
                mxe = performance(perf_pred, c("mxe"))@y.values[[1]]
                results_list[[index_r]] = data.frame(auc = perf@y.values[[1]], expr = expr, subject = subject, method = method,
                Ti = Ti, mxe = mxe)
                index_r = index_r + 1
            }

            if ("PC" %in% methods_list) {
                method = "PC"
                suffStat <- list(C = cor(fmri_data), n = nrow(fmri_data))
                results_list[[index_r]] = data.frame(auc = get_PC_AUC(suffStat, FALSE, fmri_data, true_labels), expr = expr, subject = subject, method = method,
                Ti = Ti, mxe = NA)
                index_r = index_r + 1
            }

            if ("CPC" %in% methods_list) {
                method = "CPC"
                suffStat <- list(C = cor(fmri_data), n = nrow(fmri_data))
                results_list[[index_r]] = data.frame(auc = get_PC_AUC(suffStat, TRUE, fmri_data, true_labels), expr = expr, subject = subject, method = method,
                Ti = Ti, mxe = NA)
                index_r = index_r + 1
            }

            if ("IAMB" %in% methods_list) {
                method = "IAMB"
                if (dim(fmri_data)[1] > dim(fmri_data)[2]) {
                    results_list[[index_r]] = data.frame(auc = get_AUC_bnlearn(fmri_data, true_labels, iamb), expr = expr, subject = subject, method = method,
                    Ti = Ti, mxe = NA)
                }else {

                    results_list[[index_r]] = data.frame(auc = NA, expr = expr, subject = subject, method = method,
                    Ti = Ti, mxe = NA)
                }
                index_r = index_r + 1
            }

            if ("LiNGAM" %in% methods_list) {
                method = "LiNGAM"
                if (Ti >= 100)
                {
                    R = LINGAM(fmri_data)
                    pred_net = t(abs(R$B))
                    pred = pred_net[upper.tri(pred_net) | lower.tri(pred_net)]
                    perf_pred = prediction(pred, true_labels)
                    perf = performance(perf_pred, "auc")
                    results_list[[index_r]] = data.frame(auc = perf@y.values[[1]], expr = expr, subject = subject, method = method,
                    Ti = Ti, mxe = NA)
                }else {
                    results_list[[index_r]] = data.frame(auc = NA, expr = expr, subject = subject, method = method,
                    Ti = Ti, mxe = NA)
                }
                index_r = index_r + 1
            }

            if ("PW-LiNGAM" %in% methods_list) {
                method = "PW-LiNGAM"
                if (dim(fmri_data)[1] > dim(fmri_data)[2]) {
                    results_list[[index_r]] = data.frame(auc = get_AUC_pwling(fmri_data, true_labels), expr = expr, subject = subject, method = method,
                    Ti = Ti, mxe = NA)
                }else {
                    results_list[[index_r]] = data.frame(auc = NA, expr = expr, subject = subject, method = method,
                    Ti = Ti, mxe = NA)
                }
                index_r = index_r + 1
            }
        }
    }
}

results = rbindlist(results_list)
#write.csv(all_results, file = '../../nongit/fmri/all_results.csv')
results = subset(results, expr %in% c(1, 2, 3))
results = as.data.frame(results)

results$label_N = NA
results[which(results$expr == 1),]$label_N = "N=5"
results[results$expr == 2,]$label_N = "N=10"
results[results$expr == 3,]$label_N = "N=15"
results$label_N = factor(x = results$label_N,
levels = c('N=5', 'N=10', 'N=15'))

results$method = factor(x = results$method, levels = c("LATNET", "IAMB", "PC", "CPC", "LiNGAM", "PW-LiNGAM"))
#results[results$method == "iamb",]$method = "IAMB"

write.csv(x = results, file = paste(results_output_path, 'fmri_results_all.csv', sep = ''))
ggplot(results, aes(x = as.factor(Ti), y = auc, fill = method)) +
    geom_boxplot(outlier.shape = NA, size = .1) +
#  geom_linerange(aes(ymin = tpr-tpr_sd, ymax = tpr+tpr_sd)) +
    scale_fill_brewer(palette = "OrRd") +
    theme_bw() +
    scale_y_continuous(name = "AUC") +
    xlab(expression(n[i])) +
    theme(legend.direction = "horizontal", legend.position = "top", legend.box = "horizontal",
    axis.line = element_line(colour = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    text = element_text(size = 8),
    legend.title = element_blank(),
    #      axis.text.x = element_blank(),
    legend.key = element_blank(),
    panel.background = element_rect(fill = NA, color = "black")
    ) +
    facet_wrap(~ label_N, nrow = 1) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))

full_width = 13.5
full_height = 6

ggsave(filename = paste(pdf_output_path, "fmri_auc.pdf", sep = ''),
width = full_width, height = full_height, units = "cm", device = cairo_pdf)
