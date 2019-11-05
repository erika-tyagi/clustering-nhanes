rm(list = ls())

library(tidyverse)
library(GGally)
library(factoextra)
library(kableExtra)
library(clValid)
library(ggpubr)

# import data
nhanes <- read.csv('../process-raw-data/NHANES-clean.csv') %>% 
    filter(year == '2015-2016', 
           CBD071 < 1000,
           DBD895 < 1000, 
           DBD905 < 1000, 
           DBD910 < 1000) %>%  
    select(-c(year, SEQN, FSD200, DBQ301, DBQ301, DBQ700, DBQ424)) %>%
    drop_na() %>% 
    sample_n(100)

scaled <- nhanes %>% 
    scale()

# scatterplot matrix 
ggscatmat(scaled) 
ggsave('images/scatmat.png', height = 10, width = 12)

# diagnose clusterability 
clustend <- get_clust_tendency(scaled, n = nrow(scaled)-1)
clustend$hopkins_stat
clustend$plot
ggsave('images/odi.png')

# fit k-means algorithm
kmeans <- kmeans(scaled, 
                 centers = 2,
                 nstart = 15) 

# visualize clusters 
fviz_cluster(kmeans, 
             data = scaled, 
             geom = 'point', 
             show.clust.cent = FALSE, 
             main = 'Cluster plot: K-means') +
    theme_bw()
ggsave('images/clusterplot.png')

# store cluster assignments 
t_kmeans <- data.frame(as.table(kmeans$cluster))
colnames(t_kmeans)[colnames(t_kmeans)=='Freq'] <- 'assignment_kmeans'
t_kmeans$Var1 <- NULL
t <- cbind(nhanes, t_kmeans)
table(t$assignment_kmeans)

# map variable names to descriptions
varnames <- vector(mode = 'list', length = '11')
names(varnames) <- c('CBD071', 
                     'DBD895', 
                     'DBD905', 
                     'DBD910', 
                     'DR2TCALC', 
                     'DR2TFIBE', 
                     'DR2TIRON', 
                     'DR2TSFAT', 
                     'DR2TSODI', 
                     'DR2TSUGR')
varnames[[1]] <- 'During the past 30 days, how much money \n{did your family/did you} spend at supermarkets or grocery stores?'
varnames[[2]] <- 'During the past 7 days, how many meals {did you/did SP} \nget that were prepared away from home'
varnames[[3]] <- 'During the past 30 days, how often did {you/SP} \neat "ready to eat" foods from the grocery store?'
varnames[[4]] <- 'During the past 30 days, how often did you {SP} \neat frozen meals or frozen pizzas?'
varnames[[5]] <- 'Calcium (mg)'
varnames[[6]] <- 'Dietary fiber (gm)'
varnames[[7]] <- 'Iron (mg)'
varnames[[8]] <- 'Total saturated fatty acids (gm)'
varnames[[9]] <- 'Sodium (mg)'     
varnames[[10]] <- 'Total sugars (gm)'    

# cluster assignment scatterplots
plot1 <- t %>% 
    ggplot(aes(x = DR2TSFAT, y = DBD895, color = as.factor(assignment_kmeans))) + 
    geom_jitter(size = 1.0) + 
    theme_bw() +
    labs(x = varnames[['DR2TSFAT']], y = varnames[['DBD895']]) + 
    theme(legend.title = element_blank(), 
          axis.title.x = element_text(size = 8), 
          axis.title.y = element_text(size = 8)) 
plot2 <- t %>% 
    ggplot(aes(x = DR2TSFAT, y = CBD071, color = as.factor(assignment_kmeans))) + 
    geom_jitter(size = 1.0) + 
    theme_bw() +
    labs(x = varnames[['DR2TSFAT']], y = varnames[['CBD071']]) + 
    theme(legend.title = element_blank(), 
          axis.title.x = element_text(size = 8), 
          axis.title.y = element_text(size = 8)) 
plot3 <- t %>% 
    ggplot(aes(x = DR2TSFAT, y = DR2TSODI, color = as.factor(assignment_kmeans))) + 
    geom_jitter(size = 1.0) + 
    theme_bw() +
    labs(x = varnames[['DR2TSFAT']], y = varnames[['DR2TSODI']]) + 
    theme(legend.title = element_blank(), 
          axis.title.x = element_text(size = 8), 
          axis.title.y = element_text(size = 8))
plot4 <- t %>% 
    ggplot(aes(x = DR2TSFAT, y = DR2TSUGR, color = as.factor(assignment_kmeans))) + 
    geom_jitter(size = 1.0) + 
    theme_bw() +
    labs(x = varnames[['DR2TSFAT']], y = varnames[['DR2TSUGR']]) + 
    theme(legend.title = element_blank(), 
          axis.title.x = element_text(size = 8), 
          axis.title.y = element_text(size = 8))

ggarrange(plot1, plot2, plot3, plot4)
ggsave('images/scatter_clust.png', height = 8, width = 10)

# internal valiation 
internal <- clValid(scaled, 2:10,  
                    clMethods = c('kmeans'), 
                    validation = 'internal')

optimalScores(internal)