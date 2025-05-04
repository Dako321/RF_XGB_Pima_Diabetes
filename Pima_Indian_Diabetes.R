library(dplyr)
library(lattice)
library(tidyverse)
library(caret)
library(lattice)
library(recipes)
library(hardhat)
library(sparsevctrs)
library(pROC)
library(ROCR)
library(colorspace)
library(grid)
library(VIM)
library(ranger) 
library(xgboost)
library(mice)
library(caret)
library(recipes)


# Data available on kaggle:
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data

pima_data <- read.csv("diabetes.csv", header = TRUE)

pima_data$Outcome <- as.factor(pima_data$Outcome)


# Outcome Proportion
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
imp <- mice(pima_data, m = 5, method = "pmm", seed = 123)
imp

# Extract the completed dataset from imputation 1
pima_complete <- complete(imp, 1)
pima_complete

pima_complete$Insulin
pima_data$Insulin

# Imputed Data
pima_complete
pima_data <- pima_complete
pima_data



# Splitting into training and testing set
n <- nrow(pima_data)
set.seed(123)

# 80/20 split
train_indices <- sample(seq_len(n), size = floor(0.8 * n))

pima_train <- pima_data[train_indices, ]
pima_test  <- pima_data[-train_indices, ]
pima_train
pima_test 

pima_train$Outcome
pima_test$Outcome

dim(pima_train)
dim(pima_test)  


##############################################################################
# Hyperparametertuning Random Forest

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


# For better reproducibility
set.seed(123)

# Class weights train set
w_train <- sum(pima_train$Outcome != "Diabetic")/
           sum(pima_train$Outcome == "Diabetic")

# Hyperparameter grid for Random Forest
rf_grid <- expand.grid(
  mtry = c(2, 3, 4),  
  splitrule = "gini",       
  min.node.size = c(1, 2, 3, 4, 5),      
  num.trees = c(80, 100, 200, 300, 500),
  auc = NA  # Track AUC for each combination
)

rf_grid

set.seed(123)
# Custom training loop with train()
for (i in seq_len(nrow(rf_grid))) {
  set.seed(123)
  # Train model with current hyperparameters
  fit <- train(
    Outcome ~ .,
    data = pima_train,
    method = "ranger",
    metric = "ROC",                         # Optimize for AUC
    tuneGrid = data.frame(                  # Tunable parameters 
      mtry = rf_grid$mtry[i],
      splitrule = rf_grid$splitrule[i],
      min.node.size = rf_grid$min.node.size[i]
    ),
    trControl = trainControl(
      method = "cv",                              # 5-fold cross-validation
      number = 5,
      classProbs = TRUE,                          # Required for AUC
      summaryFunction = twoClassSummary,          # Compute ROC metrics
      verboseIter = FALSE
    ),
    num.trees = rf_grid$num.trees[i],
    importance = "none",                   # Disable importance for speed
    class.weights = c(1, w_train),
    seed = 123
  )
  
  # Extract best AUC from cross-validation results
  rf_grid$auc[i] <- max(fit$results$ROC)
}

# Rank models by AUC
rf_grid %>%
  arrange(desc(auc)) %>%
  head(10)




# Class weights test data
w_test <- sum(pima_test$Outcome != "Diabetic")/
          sum(pima_test$Outcome == "Diabetic")

# Final Random Forest model
final_pima_rf1 <- ranger(
  formula         = Outcome ~ ., 
  data            = pima_train, 
  num.trees       = 500,  
  mtry            = 2,
  min.node.size   = 4,
  class.weights = c(1, w_test),
  respect.unordered.factors = "order",
  seed            = 123,
)

# With class probability for AUC-ROC
final_pima_rf2 <- ranger(
  formula         = Outcome ~ ., 
  data            = pima_train, 
  num.trees       = 500,
  mtry            = 2,
  min.node.size   = 4,
  probability     = TRUE,
  class.weights = c(1, w_test),
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
preds1 <- factor(preds1, levels = c("NonDiabetic", "Diabetic"), labels = c(0,1))

# Confusion matrix
cm <- confusionMatrix(preds1, pima_test$Outcome)
cm


# AUC-ROC
preds2 <- predict(final_pima_rf2, data = pima_test)$predictions
preds2

colnames(preds2) <- c(0,1)

roc_od <- roc(pima_test$Outcome, preds2[, "1"],)
roc_od

# Plot ROC curve
par(mfrow = c(1, 1))
plot(roc_od, 
     main = "ROC Curve for Random Forest Model",
     print.auc = TRUE, 
     legacy.axes = TRUE)



##############################################################################
# Hyperparametertuning XGboost

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
  nrounds = c(50, 100, 150, 200),               
  max_depth = c(3, 4, 5),        
  eta = c(0.01, 0.05, 0.1),         
  gamma = c(0, 0.1, 0.2),         
  colsample_bytree = 1,      
  min_child_weight = c(1, 2, 3, 4, 5),        
  subsample = 1         
)


xgb_grid


# Define cross-validation settings with 10 folds
tc <- trainControl(method = "cv",                     # 5-fold crossvalidation
                   number = 5,
                   classProbs = TRUE,                 # Required for AUC
                   summaryFunction = twoClassSummary, # # Compute ROC metrics
                   verboseIter = FALSE)


# Class weights train set
w_train <- sum(pima_train$Outcome != "Diabetic")/
  sum(pima_train$Outcome == "Diabetic")

set.seed(123)
# Train XGB with set grid
xgb_model <- train(
  Outcome ~ .,
  data = pima_train,
  method = "xgbTree",       
  metric = "ROC",           
  trControl = tc,           
  tuneGrid = xgb_grid,     
  scale_pos_weight = c(1, w_train),
  seed = 123
)

print(xgb_model)

xgb_grid$auc <- xgb_model$results$ROC

# Show top models
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
  max_depth = 4,
  min_child_weight = 2,
  subsample = 1,
  colsample_bytree = 1,
  gamma = 0.1,
  lambda = 0,
  alpha = 0
)


# Train final XGBoost model
xgb.fit.final <- xgboost(
  params = params,
  data = X,
  label = Y,
  nrounds = 50,
  scale_pos_weight = c(1, w_test),
  objective = "binary:logistic",
  verbose = 0
)


xgb.fit.final

# Prediction using test set
pred3 <- predict(xgb.fit.final, X2)
pred3

# Predictions over 0.5 will be class '1'
pred3 <-  as.numeric(pred3 > 0.5)

pred3 <- as.factor(pred3)
pred3


# Confusion matrix
xgb_test <- as.factor(pima_test$Outcome)
cm2 <- confusionMatrix(pred3, xgb_test)
cm2


# AUC-ROC
preds4 <- predict(xgb.fit.final, X2)
preds4

roc_ox <- roc(xgb_test, preds4)
roc_ox


# Plot ROC curve
plot(roc_ox, 
     main = "ROC Curve for XGBoost Model",
     print.auc = TRUE, 
     legacy.axes = TRUE)













