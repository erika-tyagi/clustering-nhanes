rm(list = ls())
options(scipen=999)

library(tidyverse)
library(clValid)
library(factoextra)
library(ggfortify)
library(ggcorrplot)

# import clean data 
clean <- read.csv('process-raw-data/NHANES-clean.csv')

# limit to adult, high recall, and usual consumption respondents 
limited <- clean %>% 
    filter(RIDAGEYR > 17 & 
           DR1_300 == 2 & DR2_300 == 2 & 
           DR1DRSTZ == 1 & DR2DRSTZ == 1) %>% 
    select(-DR1_300, -DR2_300, -DR1DRSTZ, -DR2DRSTZ) %>% 
    drop_na()

# subset demographic features 
demo <- limited %>% 
    select(year, SEQN, RIDAGEYR, RIAGENDR, INDFMPIR, RIDRETH1, 
           TKCAL, TPROT, TCARB, TSUGR, TTFAT)

# subset nutrient features 
features <- limited %>% 
    select(-year, -SEQN, -RIDAGEYR, -RIAGENDR, -INDFMPIR, -RIDRETH1)

# correlation matrix 
corr <- cor(features)
ggcorrplot(corr) + 
    scale_fill_gradient2(limit = c(-0.1, 1)) +
    theme(axis.text.x = element_blank(), 
          axis.text.y = element_blank())

# identify highly correlated features 
corr_check <- function(Dataset, threshold){
    matriz_cor <- cor(Dataset)
    matriz_cor
    for (i in 1:nrow(matriz_cor)){
        correlations <-  which((abs(matriz_cor[i,i:ncol(matriz_cor)]) > threshold) & (matriz_cor[i,i:ncol(matriz_cor)] != 1))
        
        if(length(correlations)> 0){
            lapply(correlations,FUN =  function(x) (cat(paste(colnames(Dataset)[i], "with",colnames(Dataset)[x]), "\n")))
        }
    }
}

corr_check(features, 0.90)

# run PCA 
pca <- prcomp(features, 
              scale = T, 
              center = T)
summary(pca)

# screeplot
fviz_screeplot(pca, 
               geom = 'line')

# variable contributors
fviz_contrib(pca, 
             choice = 'var', 
             top = 20, 
             axis = 1) 

fviz_pca_var(pca, 
             col.var = 'contrib', 
             select.var = list(contrib = 10), 
             repel = T)

# biplot 
fviz_pca_biplot(pca, 
                label = F, 
                pointsize = 0.5)

fviz_pca_biplot(pca, 
                select.var = list(contrib = 10), 
                label = 'var', 
                repel = T, 
                pointsize = 0.5)

# get components
comps <- data.frame(pca$x[, 1:20])

# diagnose clusterability
clustend <- get_clust_tendency(comps, n = nrow(comps) - 1)
clustend$hopkins_stat
clustend$plot 

# kmeans 
set.seed(123)
kmeans <- kmeans(comps, 
                 centers = 2, 
                 nstart = 25)
summary(kmeans)

fviz_cluster(kmeans, 
             data = comps, 
             geom = 'point', 
             show.clust.cent = F, 
             main = 'Cluster Plot: K-Means')

# # validate
# clValid(comps, 2:20, 
#         clMethods = c('kmeans'), 
#         validation = 'internal')

# combine dataframes 
t_kmeans <- data.frame(as.table(kmeans$cluster))
t_kmeans <- t_kmeans %>% 
    mutate(assignment_kmeans = Freq) %>% 
    select(assignment_kmeans)
combined <- cbind(demo, comps, t_kmeans) 

# visualize 
combined %>% 
    ggplot(aes(x = TTFAT, y = TKCAL, color = as.factor(assignment_kmeans))) +
    geom_point(size = 0.1)

# summarize numerically 
combined %>% 
    group_by(assignment_kmeans) %>% 
    summarise(avg_age = median(RIDAGEYR))


