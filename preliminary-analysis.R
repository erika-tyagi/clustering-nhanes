library(tidyverse)
library(skimr)
library(GGally)
library(factoextra)
library(cluster)
library(clValid)
library(mixtools)
library(ggrepel)
library(gridExtra)
library(mclust)
library(tidyverse)
library(datasets)
library(seriation)
library(ggdendro) 
library(gridExtra)
library(skimr)
library(dendextend)

rm(list = ls())

# import data
nhanes_clean <- read.csv('process-raw-data/NHANES-clean.csv') %>% 
    filter(year == '2015-2016') %>%
    select(-c(year, SEQN, FSD200, DBQ301, DBQ301, DBQ700)) %>%
    drop_na() %>%
    scale()

# scatterplot matrix 
ggscatmat(subset) 

# diagnose clusterability 
clustend <- get_clust_tendency(subset, n = nrow(subset)-1)

# Hopkins statistic (sparse sampling)
clustend$hopkins_stat

# VAT / ODI 
clustend$plot

# fit k-means algorithm
set.seed(123)
kmeans <- kmeans(subset, 
                 centers = 3,
                 nstart = 15) 

# summarize clusters
summary(kmeans)

# visualize clusters 
fviz_cluster(kmeans, 
             data = subset, 
             geom = 'point', 
             show.clust.cent = FALSE, 
             main = 'Cluster plot: K-means') +
    theme_bw()

# store cluster assignments 
t_kmeans <- data.frame(as.table(kmeans$cluster))
colnames(t_kmeans)[colnames(t_kmeans)=='Freq'] <- 'assignment_kmeans'
t_kmeans$Var1 <- NULL
t <- cbind(subset, t_kmeans)

t %>% 
    ggplot(aes(x = CBD071, y = DR2TSODI, color = as.factor(assignment_kmeans))) + 
    geom_point() + 
    theme_bw() +
    theme(legend.title = element_blank())







