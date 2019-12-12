rm(list = ls())
options(scipen = 3)

library(tidyverse)
library(clValid)
library(factoextra)
library(ggfortify)
library(ggcorrplot)

# import clean data 
clean <- read.csv('../process-raw-data/NHANES-clean.csv')

# limit to adult, high recall, and usual consumption respondents 
limited <- clean %>% 
    filter(RIDAGEYR > 17 & 
           DR1_300 == 2 & DR2_300 == 2 & 
           DR1DRSTZ == 1 & DR2DRSTZ == 1) %>% 
    select(-DR1_300, -DR2_300, -DR1DRSTZ, -DR2DRSTZ) %>% 
    drop_na()

# subset demographic features 
demo <- limited %>% 
    select(year, SEQN, BMXBMI, BPQ020, RIDAGEYR, RIAGENDR, INDFMPIR, RIDRETH1, 
           TKCAL, TPROT, TCARB, TSUGR, TTFAT, TPOTA, TMAGN, TFOLA, TSFAT)

# subset nutrient features 
features <- limited %>% 
    select(-year, -SEQN, -RIDAGEYR, -RIAGENDR, -INDFMPIR, -RIDRETH1)

# correlation matrix 
corr <- cor(features)
ggcorrplot(corr) + 
    scale_fill_gradient2(limit = c(-1, 1)) +
    theme(axis.text.x = element_blank(), 
          axis.text.y = element_blank()) + 
    labs(fill = "Correlation")

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

corr_check(features, 0.95)

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
comps <- data.frame(pca$x[, 1:7])

# diagnose clusterability
#clustend <- get_clust_tendency(comps, n = nrow(comps) - 1)
#clustend$hopkins_stat
#clustend$plot 

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
# clValid(comps, 2:5, 
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
    ggplot(aes(x = TCARB, y = TTFAT, color = as.factor(assignment_kmeans))) +
    geom_point(size = 0.1) + 
    labs(main = 'Cluster Assignments', 
         x = 'Carbohydrate (gm) / Energy (kcal)', 
         y = 'Total fat (gm) / Energy (kcal)') + 
    theme_bw() + 
    theme(legend.title = element_blank()) 

combined %>% 
    ggplot(mapping = aes(x = as.factor(assignment_kmeans), y = RIDAGEYR)) + 
    geom_boxplot() + 
    scale_x_discrete(labels = c('1', '2')) + 
    labs(y = 'Age in years') +
    theme_bw() + 
    theme(axis.title.x = element_blank())

combined %>% 
    ggplot(mapping = aes(x = as.factor(assignment_kmeans), y = INDFMPIR)) + 
    geom_boxplot() + 
    scale_x_discrete(labels = c('1', '2')) + 
    labs(y = 'Ratio of family income to poverty') +
    theme_bw() + 
    theme(axis.title.x = element_blank())

combined %>% 
    mutate(male = case_when(RIAGENDR == 1 ~ 1, 
                            RIAGENDR != 1 ~ 0)) %>% 
    group_by(as.factor(assignment_kmeans)) %>% 
    summarise(pct_male = mean(male)) %>% 
    ggplot(aes(x = `as.factor(assignment_kmeans)`, y = pct_male)) + 
    geom_bar(stat = 'identity') + 
    labs(y = 'Proportion Male') + 
    scale_y_continuous(limits = c(0, 1.0)) + 
    theme_bw() + 
    theme(axis.title.x = element_blank())

combined %>% 
    mutate(white = case_when(RIDRETH1 == 3 ~ 1, 
                             RIDRETH1 != 3 ~ 0)) %>% 
    group_by(as.factor(assignment_kmeans)) %>% 
    summarise(pct_white = mean(white)) %>% 
    ggplot(aes(x = `as.factor(assignment_kmeans)`, y = pct_white)) + 
    geom_bar(stat = 'identity') + 
    labs(y = 'Proportion White') + 
    scale_y_continuous(limits = c(0, 1.0)) + 
    theme_bw() + 
    theme(axis.title.x = element_blank())

combined %>% 
    mutate(hispanic = case_when(RIDRETH1 > 2 ~ 1, 
                                RIDRETH1 < 3 ~ 0)) %>% 
    group_by(as.factor(assignment_kmeans)) %>% 
    summarise(pct_hispanic = mean(hispanic)) %>% 
    ggplot(aes(x = `as.factor(assignment_kmeans)`, y = pct_hispanic)) + 
    geom_bar(stat = 'identity') + 
    labs(y = 'Proportion Mexican and/or Hispanic') + 
    scale_y_continuous(limits = c(0, 1.0)) + 
    theme_bw() + 
    theme(axis.title.x = element_blank())

combined %>% 
    mutate(black = case_when(RIDRETH1 == 4 ~ 1, 
                             RIDRETH1 != 4 ~ 0)) %>% 
    group_by(as.factor(assignment_kmeans)) %>% 
    summarise(pct_black = mean(black)) %>% 
    ggplot(aes(x = `as.factor(assignment_kmeans)`, y = pct_black)) + 
    geom_bar(stat = 'identity') + 
    labs(y = 'Proportion Black') + 
    scale_y_continuous(limits = c(0, 1.0)) + 
    theme_bw() + 
    theme(axis.title.x = element_blank())

##### WRITE TO CSV ##### 
write.csv(combined %>% 
              select(year, SEQN, BMXBMI, BPQ020, RIDAGEYR, RIAGENDR, INDFMPIR, RIDRETH1, 
                     PC1, PC2, PC3, PC4, PC5, PC6, PC7, assignment_kmeans), 
          'clustered_data.csv', row.names = FALSE)

