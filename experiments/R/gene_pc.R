# this files runs PC algorithm on gene800 data

library(pcalg)

input_file = paste('../data/Spellman/gene800_Spellman.csv')

gene_data = t(read.csv(input_file, header = T)[,2:18])
suffStat <- list(C = cor(gene_data), n = nrow(gene_data))

pc.fit = pc(suffStat, indepTest = gaussCItest, p = ncol(gene_data), alpha = 0.05, 
            conservative = T, u2pd = "relaxed")

pred_net = as(pc.fit@graph, "matrix")
sum(pred_net)

write.csv(pred_net, "../../results/gene_reg/CPC/conn.csv")
