rm(list = ls())
options(scipen=999)

library(tidyverse)
library(clValid)
library(factoextra)
library(ggfortify)

# import clean data 
clean <- read.csv('process-raw-data/NHANES-clean.csv') %>% 
    select(-year, -SEQN) 

# run PCA 
pca <- prcomp(na.omit(clean), 
              scale = T, 
              center = T)
summary(pca)

# screeplot
fviz_screeplot(pca)

# biplot
autoplot(pca,
         size = 0.5, 
         loadings = T) 

# variable contributors
fviz_contrib(pca, 
             choice = 'var', 
             top = 25, 
             axis = 1) 

# get components
comps <- data.frame(pca$x[, 1:5])

# diagnose clusterability
clustend <- get_clust_tendency(comps, n = nrow(comps) - 1)
clustend$hopkins_stat
clustend$plot 

# kmeans 
set.seed(123)
kmeans <- kmeans(comps, 
                 centers = 2, 
                 nstart = 15)
summary(kmeans)

fviz_cluster(kmeans, 
             data = comps, 
             geom = 'point', 
             show.clust.cent = F, 
             main = 'Cluster Plot: K-Means')

# validate
clValid(comps, 2:5, 
        clMethods = c('kmeans', 'pam', 'model'), 
        validation = 'internal')


