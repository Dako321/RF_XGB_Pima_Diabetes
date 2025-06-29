library(dplyr)
library(lattice)
library(tidyverse)
library(caret)
library(recipes)
library(pROC)
library(ranger) 
library(xgboost)
library(mice)
library(caret)
library(recipes)

# Data available on kaggle:
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data

pima_data <- read.csv("diabetes.csv", header = TRUE)

pima_data$Outcome <- as.factor(pima_data$Outcome)


# Outcome(Diabetis) Proportion
sum(pima_data$Outcome == 1)/length(pima_data$Outcome)
sum(pima_data$Outcome == 0)/length(pima_data$Outcome)


# Missing Data
sum(pima_data$Glucose == 0)
sum(pima_data$BloodPressure == 0)
sum(pima_data$SkinThickness == 0)
sum(pima_data$Insulin == 0)
sum(pima_data$BMI == 0)


# 0 entries into NA
pima_data$Glucose[which(pima_data$Glucose == 0)] <- NA
pima_data$BloodPressure[which(pima_data$BloodPressure == 0)] <- NA
pima_data$BMI[which(pima_data$BMI == 0)] <- NA
pima_data$SkinThickness[which(pima_data$SkinThickness == 0)] <- NA
pima_data$Insulin[which(pima_data$Insulin == 0)] <- NA


# Boxplots
boxplot(pima_data$Glucose) 
boxplot(pima_data$BloodPressure)   
boxplot(pima_data$BMI)         
boxplot(pima_data$SkinThickness)   
boxplot(pima_data$Insulin)        
boxplot(pima_data$Age)
boxplot(pima_data$Pregnancies)
boxplot(pima_data$DiabetesPedigreeFunction)


# Finding the outliers of Insulin and SkinThickness
sort(pima_data$SkinThickness)
which(pima_data$SkinThickness == 99) # 580
sort(pima_data$Insulin) 
which(pima_data$Insulin == 846) #14
which(pima_data$Insulin == 744) #229
which(pima_data$Insulin == 680) #248


# Boxplots Insulin and SkinThickness
par(mfrow = c(1, 2))

boxplot(pima_data$SkinThickness, main = "SkinThickness", col = "salmon")  
highlight_idx <- c(580)
points(x = rep(1, length(highlight_idx)),  
       y = pima_data$SkinThickness[highlight_idx],            
       col = "red",                        
       pch = 1,                          
       cex = 1)   


boxplot(pima_data$Insulin, main = "Insulin", col = "skyblue")  
highlight_idx <- c(14, 229, 248)
points(x = rep(1, length(highlight_idx)),  
       y = pima_data$Insulin[highlight_idx],            
       col = "red",                        
       pch = 1,                          
       cex = 1)                         


# Outliers removal
pima_data$SkinThickness[which(pima_data$SkinThickness == 99)] <- NA
pima_data$Insulin[which(pima_data$Insulin == 846)] <- NA
pima_data$Insulin[which(pima_data$Insulin == 744)] <- NA
pima_data$Insulin[which(pima_data$Insulin == 680)] <- NA



# MICE impuation
set.seed(123)
imp <- mice(pima_data, m = 5, method = "pmm")

# Extract the completed dataset from imputation 1
pima_complete <- complete(imp, 1)

# Imputed Data
pima_data <- pima_complete


# Splitting into training and testing set
n <- nrow(pima_data)
set.seed(123)

# 80/20 split
train_indices <- sample(seq_len(n), size = floor(0.8 * n))

pima_train <- pima_data[train_indices, ]
pima_test  <- pima_data[-train_indices, ]



##############################################################################
# Hyperparametertuning Random Forest

# Convert Outcome to a factor with names(string)
pima_train$Outcome <- factor(
  pima_train$Outcome,
  levels = c(0, 1),  
  labels = c("NonDiabetic", "Diabetic") 
)

pima_test$Outcome <- factor(
  pima_test$Outcome,
  levels = c(0, 1), 
  labels = c("NonDiabetic", "Diabetic")
)

pima_train$Outcome
pima_test$Outcome


# For better reproducibility
set.seed(123)

# Class weights train set
w_train0 <- length(pima_train$Outcome)/
            (2 * sum(pima_train$Outcome != "Diabetic"))
w_train1 <- length(pima_train$Outcome)/
            (2 * sum(pima_train$Outcome == "Diabetic"))



# Hyperparameter grid for Random Forest
rf_grid <- expand.grid(
  mtry = c(2, 3, 4),  
  splitrule = "gini",       
  min.node.size = c(5, 10, 15, 20, 25, 30),      
  num.trees = c(80, 100, 200, 300),
  auc = NA  # Track AUC for each combination
)

rf_grid



set.seed(123)
# Loop with train()
for (i in seq_len(nrow(rf_grid))) {
  set.seed(123)
  # Train model with current hyperparameters
  fit <- train(
    Outcome ~ .,
    data = pima_train,
    method = "ranger",
    metric = "ROC",    # Optimize for AUC
    tuneGrid = data.frame(                  
      mtry = rf_grid$mtry[i],
      splitrule = rf_grid$splitrule[i],
      min.node.size = rf_grid$min.node.size[i]
    ),
    trControl = trainControl(
      method = "cv",  # 5-fold cross-validation                     
      number = 5,
      classProbs = TRUE,                         
      summaryFunction = twoClassSummary,        
      verboseIter = FALSE
    ),
    num.trees = rf_grid$num.trees[i],
    importance = "none",                   
    class.weights = c(w_train0, w_train1),
    seed = 123
  )
  
  # Best AUC from cross-validation results
  rf_grid$auc[i] <- max(fit$results$ROC)
}

# Rank models by AUC
rf_grid %>%
  arrange(desc(auc)) %>%
  head(10)



# Class weights test data
w_test0 <- length(pima_test$Outcome)/
            (2 * sum(pima_test$Outcome != "Diabetic"))
w_test1 <- length(pima_test$Outcome)/
            (2 * sum(pima_test$Outcome == "Diabetic"))


# Final Random Forest model
final_pima_rf1 <- ranger(
  formula         = Outcome ~ ., 
  data            = pima_train, 
  num.trees       = 200,  
  mtry            = 2,
  min.node.size   = 30,
  class.weights = c(w_test0, w_test1),
  respect.unordered.factors = "order",
  seed            = 123,
)

# With class probability for AUC-ROC
final_pima_rf2 <- ranger(
  formula         = Outcome ~ ., 
  data            = pima_train, 
  num.trees       = 200,
  mtry            = 2,
  min.node.size   = 30,
  probability     = TRUE,
  class.weights = c(w_test0, w_test1),
  seed            = 123,
)


final_pima_rf1
final_pima_rf2



# Using orginal data with orginal labels for Outcome
pima_train <- pima_data[train_indices, ]
pima_test  <- pima_data[-train_indices, ]


# Prediction using test set
preds1 <- predict(final_pima_rf1, data = pima_test)$predictions

# Using orginal labels
preds1 <- factor(preds1, levels = c("NonDiabetic", "Diabetic"), 
                 labels = c(0,1))

# Confusion matrix test set
cm <- confusionMatrix(preds1, pima_test$Outcome)
cm

# F1-score of test set
precision <- cm$byClass[3][1]    #
recall <- cm$byClass[3][1]       
F1 <- 2 * (precision * recall) / (precision + recall)
names(F1) <- "F1"
F1



# Prediction using train set
preds2 <- predict(final_pima_rf1, data = pima_train)$predictions

# Using orginal labels
preds2 <- factor(preds2, levels = c("NonDiabetic", "Diabetic"), 
                 labels = c(0,1))

# Confusion matrix train set
cm2 <- confusionMatrix(preds2, pima_train$Outcome)
cm2



# AUC-ROC using class probability
preds_auc <- predict(final_pima_rf2, data = pima_test)$predictions
preds_auc

colnames(preds_auc) <- c(0,1)

roc_od <- roc(pima_test$Outcome, preds_auc[, "1"],)
roc_od

# Plot ROC curve
par(mfrow = c(1, 1))
plot(roc_od, 
     main = "ROC Curve for Random Forest Model",
     print.auc = TRUE, 
     legacy.axes = TRUE)





##############################################################################
# Hyperparametertuning XGboost

pima_train <- pima_data[train_indices, ]
pima_test  <- pima_data[-train_indices, ]

# For better reproducibility
set.seed(123)

# Convert Outcome to a factor with names
pima_train$Outcome <- factor(
  pima_train$Outcome,
  levels = c(0, 1),  
  labels = c("NonDiabetic", "Diabetic") 
)

pima_test$Outcome <- factor(
  pima_test$Outcome,
  levels = c(0, 1), 
  labels = c("NonDiabetic", "Diabetic")
)

pima_train$Outcome
pima_test$Outcome


# Hyperparameter grid for XGBoost 
xgb_grid <- expand.grid(
  nrounds = c(50, 100, 150),           
  max_depth = c(2, 3, 4),        
  eta = c(0.01, 0.05, 0.1),         
  gamma = c(0, 0.1, 0.2),         
  colsample_bytree = 1,      
  min_child_weight = c(5, 10, 15, 20),        
  subsample = 1         
)


xgb_grid


# Cross-validation settings with 5 folds
tc <- trainControl(method = "cv",                     
                   number = 5,
                   classProbs = TRUE,                
                   summaryFunction = twoClassSummary, 
                   verboseIter = FALSE)


# Class weights train set
w_train0 <- length(pima_train$Outcome)/
                  (2 * sum(pima_train$Outcome != "Diabetic"))
w_train1 <- length(pima_train$Outcome)/
                  (2 * sum(pima_train$Outcome == "Diabetic"))

w_train <- w_train1/w_train0


# Class weights test data
w_test0 <- length(pima_test$Outcome)/
                 (2 * sum(pima_test$Outcome != "Diabetic"))
w_test1 <- length(pima_test$Outcome)/
                 (2 * sum(pima_test$Outcome == "Diabetic"))

w_test <- w_test1/w_test0



set.seed(123)
# Train XGB with set grid
xgb_model <- train(
  Outcome ~ .,
  data = pima_train,
  method = "xgbTree",       
  metric = "ROC",           
  trControl = tc,           
  tuneGrid = xgb_grid,     
  scale_pos_weight = w_train,
  seed = 123
)

xgb_grid$auc <- xgb_model$results$ROC

# Rank models by AUC
xgb_grid %>%
  arrange(desc(auc)) %>%
  head(10)




# Original data
pima_train <- pima_data[train_indices, ]
pima_test  <- pima_data[-train_indices, ]


# Convert to numeric
pima_train$Outcome <- as.numeric(as.character(pima_train$Outcome))
pima_train <- pima_train[sample(nrow(pima_train)), ]

pima_test$Outcome <- as.numeric(as.character(pima_test$Outcome))
pima_test <- pima_test[sample(nrow(pima_test)), ]



# Data preperation for XGBoost 

# Training set
xgb_prep <- recipe(Outcome ~ ., data = pima_train) %>%
  step_integer(all_nominal()) %>%
  prep(training = pima_train, retain = TRUE) %>%
  juice()

X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "Outcome")])
Y <- xgb_prep$Outcome
X
Y

# Testing set
xgb_prep2 <- recipe(Outcome ~ ., data = pima_test) %>%
  step_integer(all_nominal()) %>%
  prep(training = pima_test, retain = TRUE) %>%
  juice()

X2 <- as.matrix(xgb_prep2[setdiff(names(xgb_prep2), "Outcome")])
Y2 <- xgb_prep2$Outcome
X2
Y2


# Optimal parameter list
params <- list(
  eta = 0.1,
  max_depth = 3,
  min_child_weight = 20,
  subsample = 1,
  colsample_bytree = 1,
  gamma = 0.1,
  lambda = 0,
  alpha = 0
)

set.seed(123)
# Train final XGBoost model
xgb.fit.final <- xgboost(
  params = params,
  data = X,
  label = Y,
  nrounds = 50,
  scale_pos_weight = w_test,
  objective = "binary:logistic",
  verbose = 0
)


xgb.fit.final


# Prediction using test set
pred3_p <- predict(xgb.fit.final, X2)
pred3_p

# Predictions over 0.5 will be class '1' for balanced data
pred3 <-  as.numeric(pred3_p > 0.5)

pred3 <- as.factor(pred3)
pred3


# Confusion matrix test set
xgb_test <- as.factor(pima_test$Outcome)
cm3 <- confusionMatrix(pred3, xgb_test)
cm3


# F1-score of test set
precision <- cm3$byClass[3][1]    
recall <- cm3$byClass[3][1]  
F1 <- 2 * (precision * recall) / (precision + recall)
names(F1) <- "F1"
F1



# Prediction using train set
pred4 <- predict(xgb.fit.final, X)
pred4

# Predictions over 0.5 will be class '1' for balanced data
pred4 <-  as.numeric(pred4 > 0.5)

pred4 <- as.factor(pred4)
pred4

# Confusion matrix
xgb_train <- as.factor(pima_train$Outcome)
cm4 <- confusionMatrix(pred4, xgb_train)
cm4



# AUC-ROC test
roc_ox3 <- roc(xgb_test, pred3_p)
roc_ox3



# Plot ROC curve XGBoost
plot(roc_ox3, 
     main = "ROC Curves for XGBoost Model",
     print.auc = TRUE, 
     legacy.axes = TRUE)


# Plot ROC curve both models
plot(roc_ox3, 
     main = "ROC Curves: Random Forest vs XGBoost",
     legacy.axes = TRUE, col = "blue")


# Add the second ROC curve to the plot
lines(roc_od, col = "red", lwd = 2)

legend("bottomright",
       legend = c(paste0("Random Forest (AUC = ", round(auc(roc_od), 3),")"),
                  paste0("XGBoost (AUC = ", round(auc(roc_ox3), 3), ")")),
       col = c("blue", "red"),
       lwd = 2)













