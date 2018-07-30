library(bnlearn)
library(igraph)
library(pcalg)
library(textir)
library(Cairo)

pdf_output_path = '../../nongit/graphs/'
latnet_add = '../../nongit/archive/NSW_house/'
input_price_data = '../data/NSW_house/data.csv'

get_netplot = function(links, title=NA, fileoutput=NA){
    net2 <- graph.adjacency(links)

    if (! is.na(fileoutput)) {
        write_graph(net2, fileoutput, format = "edgelist")
    }
    #l <- layout.kamada.kawai(net2)
    #l <- as.matrix(lon_lats[,c("lon", "lat")] + 100)
    #  l <- as.matrix(lon_lats[,c("lon", "lat")])
    l <- as.matrix(pos_rank)

    plot(net2, layout = l,
    vertex.color = colrs, layout = l , vertex.size = 5,
    edge.arrow.size = .2, main = title, vertex.label = NA, vertex.frame.color = NA
    )
}

library(geosphere)
distance_between = function(links){
    conns = which(links, arr.ind = T)
    if (nrow(conns) == 0) {
        c()
    }else {
        x1 = lon_lats[conns[, "row"],][, c("lon", "lat")]
        x2 = lon_lats[conns[, "col"],][, c("lon", "lat")]
        d = c()
        for (i in c(1 : nrow(x1))) {
            d = c(d, distm (x1[i,], x2[i,], fun = distHaversine))
        }
        d / 1000
    }
}



full_width = 16
full_height = 16

lon_lats = read.csv('../data/NSW_house/long_lat.csv')

pos_rank = data.frame(rank(lon_lats[, c("lon")]), rank(lon_lats[, c("lat")]))

colrs = c(rep(c("red"), 11), rep(c("green"), 15),
rep(c("blue"), 17), rep(c("yellow"), 8))

names = paste(1995 : 2010, '_', 1999 : 2014, sep = '')
link_prop_nodiag_all = data.frame()

# for latnet
var_name = 'p'
latnet_thresh = 0.597
total_links = NA
sum_p = NA
avs_links = 0
CairoPDF(paste(pdf_output_path, 'NSW_house_output_latnet.pdf', sep = ""), full_width / 2.5, full_height / 2.5, bg = "transparent")
par(mfrow = c(4, 4), mar = c(0, 0, 1, 0))
total_distance = c()
for (n in names) {
    data = read.csv(paste(latnet_add, n, '/', var_name, '.csv', sep = ''), header = F)
    links = abs(t(data)) > latnet_thresh
    if (is.na(total_links)) {
        total_links = links
        sum_p = abs(t(data))
    }else {
        total_links = total_links + links
        sum_p = sum_p + abs(t(data))
    }
    total_distance = c(total_distance, distance_between(links))
    avs_links = avs_links + sum(links, na.rm = T)
    get_netplot(links, paste(substr(n, 1, 4), substr(n, 6, 10)), NA)
}
latnet_total_distance = total_distance
link_prop = total_links / length(names)
link_se = sqrt(link_prop * (1 - link_prop) / length(names))
low_link_CI = link_prop - 1.96 * link_se
high_link_CI = link_prop + 1.96 * link_se
sig_links_count = (sum((low_link_CI) > 0) * 1)
print(avs_links / length(names))
print(sum(total_distance) / avs_links)
print(sig_links_count / sum(total_links != 0))
dev.off()

link_prop_nodiag = link_prop[upper.tri(link_prop) | lower.tri(link_prop)]
link_prop_nodiag = link_prop_nodiag[link_prop_nodiag != 0]
link_prop_nodiag = data.frame(method = "LATNET", x = link_prop_nodiag)
link_prop_nodiag_all = rbind(link_prop_nodiag, link_prop_nodiag_all)

### procesing latnet output
maxs = sort(link_prop, decreasing = T)[1 : 10]
which(link_prop > maxs[9], arr.ind = TRUE)
which(link_prop == maxs[1], arr.ind = TRUE)

### procesing latnet output
maxs = sort(sum_p, decreasing = T)
which(sum_p > maxs[15], arr.ind = TRUE)
which(link_prop == maxs[1], arr.ind = TRUE)


# for pc
avs_links = 0
total_links = NA
pc_thresh = 0.015
window_size = 4
window_slide = 1
start_pose = 1995
total_distance = c()


CairoPDF(paste(pdf_output_path, 'NSW_house_output_pc.pdf', sep = ""), full_width / 2.5, full_height / 2.5, bg = "transparent")
par(mfrow = c(4, 4), mar = c(0, 0, 1, 0))

end_pose = start_pose + window_size
while (end_pose <= 2014) {
from_ = start_pose
to_ = end_pose
price_data = t(read.csv(input_price_data)
[, c(-1)])[(4*(from_ -1995) + 1):(4*(to_ - 1999) + 20),]
suffStat <- list(C = cor(price_data), n = nrow(price_data))
pc.fit = pc(suffStat, indepTest = gaussCItest, p = ncol(price_data), alpha = pc_thresh,
conservative = F, u2pd = "relaxed")
links = as(pc.fit@graph, "matrix")
if(is.na(total_links)){
total_links = links
}else {
total_links = total_links + links
}
total_distance = c(total_distance, distance_between(links == 1))
avs_links = avs_links + sum(links, na.rm = T)
get_netplot(links, paste(from_, to_), NA)
start_pose = start_pose + window_slide
end_pose = start_pose + window_size
}
pc_total_distance = total_distance
link_prop = total_links  / length(names)
link_se = sqrt(link_prop * (1 - link_prop) / length(names))
low_link_CI = link_prop - 1.96 * link_se
high_link_CI = link_prop + 1.96 * link_se
sig_links_count = (sum((low_link_CI) > 0) * 1)
print(avs_links / length(names))
print(sum(total_distance) / avs_links)
print(sig_links_count / sum(total_links != 0))
dev.off()

link_prop_nodiag = link_prop[upper.tri(link_prop) | lower.tri(link_prop)]
link_prop_nodiag = link_prop_nodiag[link_prop_nodiag != 0]
link_prop_nodiag = data.frame(method = "PC", x = link_prop_nodiag)
link_prop_nodiag_all = rbind(link_prop_nodiag, link_prop_nodiag_all)


#for CPC
avs_links = 0
total_links = NA
pc_thresh = 0.012
window_size = 4
window_slide = 1
start_pose = 1995

CairoPDF(paste(pdf_output_path, 'NSW_house_output_cpc.pdf', sep = ""), full_width/2.5, full_height/2.5, bg="transparent")
par(mfrow=c(4, 4), mar=c(0, 0, 1, 0))
total_distance = c()

end_pose = start_pose + window_size
while(end_pose <= 2014){
from_ = start_pose
to_ = end_pose
price_data = t(read.csv(input_price_data)
[, c(-1)])[(4*(from_ -1995) + 1):(4*(to_ - 1999) + 20),]
suffStat <- list(C = cor(price_data), n = nrow(price_data))
pc.fit = pc(suffStat, indepTest = gaussCItest, p = ncol(price_data), alpha = pc_thresh,
conservative = T, u2pd = "relaxed")
links = as(pc.fit@graph, "matrix")
if(is.na(total_links)){
total_links = links
}else {
total_links = total_links + links
}
total_distance = c(total_distance, distance_between(links == 1))
avs_links = avs_links + sum(links, na.rm = T)
get_netplot(links, paste(from_, to_), NA)
start_pose = start_pose + window_slide
end_pose = start_pose + window_size
}
cpc_total_distance = total_distance
link_prop = total_links  / length(names)
link_se = sqrt(link_prop * (1 - link_prop) / length(names))
low_link_CI = link_prop - 1.96 * link_se
high_link_CI = link_prop + 1.96 * link_se
sig_links_count = (sum((low_link_CI) > 0) * 1)
print(avs_links / length(names))
print(sum(total_distance) / avs_links)
print(sig_links_count / sum(total_links != 0))
dev.off()

link_prop_nodiag = link_prop[upper.tri(link_prop) | lower.tri(link_prop)]
link_prop_nodiag = link_prop_nodiag[link_prop_nodiag != 0]
link_prop_nodiag = data.frame(method = "CPC", x = link_prop_nodiag)
link_prop_nodiag_all = rbind(link_prop_nodiag, link_prop_nodiag_all)

# for PW-LiNGAM
avs_links = 0
total_links = NA
pw_thresh = 0.5
window_size = 4
window_slide = 1
start_pose = 1995
total_distance = c()

CairoPDF(paste(pdf_output_path, 'NSW_house_output_pw_lingam.pdf', sep = ""), full_width/2.5, full_height/2.5, bg="transparent")
par(mfrow=c(4, 4), mar=c(0, 0, 1, 0))

end_pose = start_pose + window_size
require(ppcor)
source('pwling.R')
while(end_pose <= 2014){
print(end_pose)
from_ = start_pose
to_ = end_pose
price_data = t(read.csv(input_price_data)
[, c(-1)])[(4*(from_ -1995) + 1):(4*(to_ - 1999) + 20),]
p_value = pcor(price_data, method = "kendall")$estimate
pred_net = abs(p_value) > pw_thresh
direc = pwling(t(as.matrix(price_data)))
links = pred_net * direc > 0
if(is.na(total_links)){
total_links = links
}else {
total_links = total_links + links
}
total_distance = c(total_distance, distance_between(links == 1))
avs_links = avs_links + sum(links, na.rm = T)
get_netplot(links, paste(from_, to_), NA)
start_pose = start_pose + window_slide
end_pose = start_pose + window_size
}
pw_ling_total_distance = total_distance
link_prop = total_links  / length(names)
link_se = sqrt(link_prop * (1 - link_prop) / length(names))
low_link_CI = link_prop - 1.96 * link_se
high_link_CI = link_prop + 1.96 * link_se
sig_links_count = (sum((low_link_CI) > 0) * 1)
print(sum(total_distance) / avs_links)
print(avs_links / length(names))
print(sig_links_count / sum(total_links != 0))
dev.off()

link_prop_nodiag = link_prop[upper.tri(link_prop) | lower.tri(link_prop)]
link_prop_nodiag = link_prop_nodiag[link_prop_nodiag != 0]
link_prop_nodiag = data.frame(method = "PW-LiNGAM", x = link_prop_nodiag)
link_prop_nodiag_all = rbind(link_prop_nodiag, link_prop_nodiag_all)


# for iamb
iamb_thresh = 0.0000001
window_size = 4
window_slide = 1
start_pose = 1995
avs_links = 0
total_links = NA
total_distance = c()

CairoPDF(paste(pdf_output_path, 'NSW_house_output_iamb.pdf', sep = ""), full_width/2.5, full_height/2.5, bg="transparent")
par(mfrow=c(4, 4), mar=c(0, 0, 1, 0))

end_pose = start_pose + window_size
while(end_pose <= 2014){
from_ = start_pose
to_ = end_pose
price_data = t(read.csv(input_price_data)
[, c(-1)])[(4*(from_ -1995) + 1):(4*(to_ - 1999) + 20),]
bn_fit = iamb(as.data.frame(price_data), alpha =  iamb_thresh)
links = amat(bn_fit)
avs_links = avs_links + sum(links, na.rm = T)
if(is.na(total_links)){
total_links = links
}else {
total_links = total_links + links
}
total_distance = c(total_distance, distance_between(links == 1))
get_netplot(links, paste(from_, to_), NA)
start_pose = start_pose + window_slide
end_pose = start_pose + window_size
}
iamb_total_distance = total_distance
link_prop = total_links  / length(names)
link_se = sqrt(link_prop * (1 - link_prop) / length(names))
low_link_CI = link_prop - 1.96 * link_se
high_link_CI = link_prop + 1.96 * link_se
sig_links_count = (sum((low_link_CI) > 0) * 1)
print(sum(total_distance) / avs_links)
print(avs_links / length(names))
print(sig_links_count / sum(total_links != 0))
dev.off()

link_prop_nodiag = link_prop[upper.tri(link_prop) | lower.tri(link_prop)]
link_prop_nodiag = link_prop_nodiag[link_prop_nodiag != 0]
link_prop_nodiag = data.frame(method = "IAMB", x = link_prop_nodiag)
link_prop_nodiag_all = rbind(link_prop_nodiag, link_prop_nodiag_all)

## test of difference between distances
t.test(latnet_total_distance, pc_total_distance)
t.test(latnet_total_distance, cpc_total_distance)
t.test(latnet_total_distance, iamb_total_distance)
t.test(latnet_total_distance, pw_ling_total_distance)

dpc_total_distance = data.frame(method = "PC", dist = pc_total_distance)
dcpc_total_distance = data.frame(method = "CPC", dist = cpc_total_distance)
diamb_total_distance = data.frame(method = "IAMB", dist = iamb_total_distance)
dpw_ling_total_distance = data.frame(method = "PW-LiNGAM", dist = pw_ling_total_distance)
dlatnet_total_distance = data.frame(method = "LATNET", dist = latnet_total_distance)

ddist = rbind (dpc_total_distance, dcpc_total_distance, diamb_total_distance, dpw_ling_total_distance, dlatnet_total_distance)
ddist$method = factor(ddist$method, levels= c("LATNET", "CPC", "IAMB", "PC", "PW-LiNGAM"))

library(ggplot2)
ggplot(ddist, aes(y = dist, x=method, fill=method)) +
geom_boxplot(size = .2, outlier.size = 0.1)+
#  geom_jitter(alpha=0.1, size = 0.01)+
theme_bw() +
scale_fill_brewer(palette="OrRd") +
xlab("") +
ylab("distance (km)") +
theme(legend.direction = "horizontal", legend.position = "none", legend.box = "horizontal",
axis.line = element_line(colour = "black"),
panel.grid.major=element_blank(),
axis.line.y=element_blank(),
panel.grid.minor=element_blank(),
#        panel.border = element_blank(),
text=element_text(family="Arial", size=10),
plot.margin = unit(x = c(0.01, 0.01, 0.01, 0.01), units = "cm"),
legend.title=element_blank(),
axis.text.x = element_text(angle = 45, hjust = 1),
legend.key = element_blank(),
panel.background = element_rect(fill=NA, color ="black")
) +
guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))

ggsave(filename = paste(pdf_output_path, "distance.pdf", sep = ''),
width = 3.5, height = 6, units = "cm", device=cairo_pdf)


## significance level
n = length(names)
sig_lev = (1.96)^2 / (n  + (1.96)^2)

library(ggplot2)
link_prop_nodiag_all$method = factor(link_prop_nodiag_all$method, levels= c("LATNET", "CPC", "IAMB", "PC", "PW-LiNGAM"))
# plot density
ggplot(link_prop_nodiag_all, aes(y = x, x=method, fill=method)) +
geom_rect(xmin=-Inf, xmax=Inf, ymin=sig_lev, ymax=Inf, alpha=0.01, fill="gray") +
geom_boxplot(size = .2, outlier.size = 0.1)+
#  geom_jitter(alpha=0.2, size = 0.01)+
#  annotate("text", x = 3, y = 0.9, label = "r statistically > 0")+
theme_bw() +
scale_fill_brewer(palette="OrRd") +
xlab("") +
ylab("r") +
ylim(c(0, 1)) +
theme(legend.direction = "horizontal", legend.position = "none", legend.box = "horizontal",
axis.line = element_line(colour = "black"),
panel.grid.major=element_blank(),
axis.line.y=element_blank(),
panel.grid.minor=element_blank(),
#        panel.border = element_blank(),
text=element_text(family="Arial", size=10),
legend.title=element_blank(),
plot.margin = unit(x = c(0.01, 0.01, 0.01, 0.01), units = "cm"),
axis.text.x = element_text(angle = 45, hjust = 1),
legend.key = element_blank(),
panel.background = element_rect(fill=NA, color ="black")
) +
guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))

ggsave(filename = paste(pdf_output_path, "prices_r.pdf", sep = ''),
width = 3.5, height = 6, units = "cm", device=cairo_pdf)


# generating all graphs
CairoPDF(paste(pdf_output_path, 'NSW_house_2010_2014_latnet.pdf', sep = ""), (full_width * 1 /6)/2.5, 3/2.5, bg="transparent")
par(mfrow=c(1, 1), mar=c(0, 0, 0, 0))
n = "2010_2014"
data = read.csv(paste(latnet_add, n, '/', var_name, '.csv', sep=''), header = F)
links = abs(t(data)) > latnet_thresh
get_netplot(links, NA, NA)
dev.off()

from_ = 2010
to_ = 2014
price_data = t(read.csv(input_price_data)
[, c(-1)])[(4*(from_ -1995) + 1):(4*(to_ - 1999) + 20),]
CairoPDF(paste(pdf_output_path, 'NSW_house_2010_2014_pc.pdf', sep = ""), (full_width * 1 /6)/2.5, 3/2.5, bg="transparent")
par(mfrow=c(1, 1), mar=c(0, 0, 0, 0))
suffStat <- list(C = cor(price_data), n = nrow(price_data))
pc.fit = pc(suffStat, indepTest = gaussCItest, p = ncol(price_data), alpha = pc_thresh, conservative = F, u2pd = "relaxed")
links = as(pc.fit@graph, "matrix")
get_netplot(links, NA, NA)
dev.off()


CairoPDF(paste(pdf_output_path, 'NSW_house_2010_2014_cpc.pdf', sep = ""), (full_width * 1 /6)/2.5, 3/2.5, bg="transparent")
par(mfrow=c(1, 1), mar=c(0, 0, 0, 0))
suffStat <- list(C = cor(price_data), n = nrow(price_data))
pc.fit = pc(suffStat, indepTest = gaussCItest, p = ncol(price_data), alpha = pc_thresh, conservative = T, u2pd = "relaxed")
links = as(pc.fit@graph, "matrix")
get_netplot(links, NA, NA)
dev.off()


CairoPDF(paste(pdf_output_path, 'NSW_house_2010_2014_iamb.pdf', sep = ""), (full_width * 1 /6)/2.5, 3/2.5, bg="transparent")
par(mfrow=c(1, 1), mar=c(0, 0, 0, 0))
bn_fit = iamb(as.data.frame(price_data), alpha =  iamb_thresh)
links = amat(bn_fit)
get_netplot(links, NA, NA)
start_pose = start_pose + window_slide
end_pose = start_pose + window_size
dev.off()


CairoPDF(paste(pdf_output_path, 'NSW_house_2010_2014_pwlingam.pdf', sep = ""), (full_width * 1 /6)/2.5, 3/2.5, bg="transparent")
par(mfrow=c(1, 1), mar=c(0, 0, 0, 0))
p_value = pcor(price_data, method = "kendall")$estimate
pred_net = abs(p_value) > pw_thresh
direc = pwling(t(as.matrix(price_data)))
links = pred_net * direc > 0
get_netplot(links, NA, NA)
dev.off()


#library(ggmap)
#subsurbs = paste(read.csv('../../data/NSW_house/data.csv')[,1], ",NSW, Australia")
# long_lat = geocode(subsurbs)
# write.csv(long_lat, '../../data/NSW_house/long_lat.csv')


# thresh = 0.55
# pdf('../../../results/NSW_house/output_latne_tmp1.pdf')
# data = read.csv('../../../results/NSW_house/tmp1/p.csv', header = F)
# links = abs(t(data)) > thresh
# get_netplot(links, "2000-2014")
# dev.off()

# thresh = 0.25
# par(mfrow=c(1, length(names))) 
# pdf('../../../results/NSW_house/output_latnet_same.pdf')
# for (n in names){
#   
#   mu = read.csv(paste('../../../results/NSW_house/full_sigmag2/', n, '/', 'mu', '.csv', sep=''), header = F)
#   sigma2 = read.csv(paste('../../../results/NSW_house/full_sigmag2/', n, '/', 'sigma2', '.csv', sep=''), header = F)
#   p = read.csv(paste('../../../results/NSW_house/full_sigmag2/', n, '/', 'p', '.csv', sep=''), header = F)
#   
#   data = abs(mu) - sqrt(sigma2) > 0
#   
# #   data = matrix(nrow = nrow(mu), ncol = ncol(mu))
# #   for (i in 1:nrow(data)){
# #     for(j in 1:ncol(data))
# #       data[j,i] = p[i,j] * pnorm(0, mu[i,j], sqrt(sigma2[i,j]))
# #   }
#   
#   links = abs(data) > thresh
#   get_netplot(links, n, paste('../../../results/NSW_house/', n, '.csv', sep=''))
# }
# dev.off()