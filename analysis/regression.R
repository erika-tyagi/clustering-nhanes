rm(list = ls())
options(scipen = 3)

library(tidyverse)
library(stargazer)

# import clustered data 
clustered <- read.csv('clustered_data.csv')

# create dependent variables  
clustered <- clustered %>% 
    mutate(is_obese = case_when(BMXBMI > 30 ~ 1, 
                                BMXBMI <= 30 ~ 0), 
           has_high_bp = case_when(BPQ020 == 1 ~ 1, 
                                   BPQ020 != 1 ~ 0)
    ) 

##### OBESITY REGRESSIONS ##### 

# just demographics 
model1 <- lm(is_obese ~ RIDAGEYR + as.factor(RIAGENDR) + INDFMPIR + as.factor(RIDRETH1),  
             data = clustered) 
summary(model1)

# demographics and clusters 
model2 <- lm(is_obese ~ RIDAGEYR + as.factor(RIAGENDR) + INDFMPIR + as.factor(RIDRETH1) 
             + as.factor(assignment_kmeans),  
             data = clustered) 
summary(model2)

# demographics and components
model3 <- lm(is_obese ~ RIDAGEYR + as.factor(RIAGENDR) + INDFMPIR + as.factor(RIDRETH1) 
             + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7,  
             data = clustered) 
summary(model3)

# compare all three models 
stargazer(model1, model2, model3, 
          type = 'text', 
          omit.stat = c('f', 'ser'))

##### HIGH BLOOD PRESSURE REGRESSIONS ##### 

# just demographics 
model4 <- lm(has_high_bp ~ RIDAGEYR + as.factor(RIAGENDR) + INDFMPIR + as.factor(RIDRETH1),  
             data = clustered) 
summary(model4)

# demographics and clusters 
model5 <- lm(has_high_bp ~ RIDAGEYR + as.factor(RIAGENDR) + INDFMPIR + as.factor(RIDRETH1) 
             + as.factor(assignment_kmeans),  
             data = clustered) 
summary(model5)

# demographics and components
model6 <- lm(has_high_bp ~ RIDAGEYR + as.factor(RIAGENDR) + INDFMPIR + as.factor(RIDRETH1) 
             + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7,  
             data = clustered) 
summary(model6)

# compare all three models 
stargazer(model4, model5, model6, 
          type = 'text', 
          omit.stat = c('f', 'ser'))

