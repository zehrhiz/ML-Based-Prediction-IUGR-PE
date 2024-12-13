########### Machine learning based Feature selection for Gene Expression data
############ for IUGR and preeclampsia Classification ###########

#### Loading the required libraries ####
library(caret)
library(data.table)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(Boruta)
library(pROC)
library(purrr)
library(randomForest)

################ Download data into local machine from link: 
################ https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE114691

#### Path to the .tsv.gz file ####
file_path <- "path/to/your/file.tsv.gz" ##### Copy/paste file path here

#### Read the compressed TSV file ####
data <- read.delim(gzfile(file_path), sep = "\t")

data_t <- data.table::transpose(data,
                                keep.names = "RefID", 
                                make.names = "GeneID")

#### Define group vector #####
target_column <- c(
  rep("Control", 21),   # 21 Control samples
  rep("PE", 20),        # 20 PE samples
  rep("PEIUGR", 20),    # 20 PEIUGR samples
  rep("IUGR", 18)       # 18 IUGR samples
)
#### Add target column to dataset & remove RefID column #####
data_t <- cbind(Target = target_column, data_t[, -1])

##### Create data subsets #####
subset_1 <- data_t[1:41,] #### control vs PE
subset_2 <- data_t[c(1:21,42:61),] #### control vs PEIUGR
subset_3 <- data_t[c(1:21,62:79),] #### control vs IUGR
subset_4 <- data_t[c(22:41,62:79),] #### IUGR vs PE

###### Convert target column to numeric
subset_1$Target <- ifelse(subset_1$Target == "Control", 0, 1)
subset_2$Target <- ifelse(subset_2$Target == "Control", 0, 1)
subset_3$Target <- ifelse(subset_3$Target == "Control", 0, 1)
subset_4$Target <- ifelse(subset_4$Target == "IUGR", 0, 1)

####################################################################################
# List of datasets
datasets <- list(subset_1, subset_2, subset_3, subset_4)
dataset_names <- c("subset_1", "subset_2", "subset_3", "subset_4")

################### Function to apply the pipeline #######################
apply_pipeline <- function(data, dataset_name) {
 
  #### Step 1: First Splitting 70/30 ####
  index_1 <- caret::createDataPartition(data$Target, p = 0.7, list = FALSE)
  Train_1 <- data[index_1, ]
  Test_1 <- data[-index_1, ]
  
  Train_target_1 <- as.factor(Train_1$Target)
  Test_target_1 <- as.factor(Test_1$Target)
  Train_genes_1 <- as.data.frame(Train_1[, -1])
  Test_genes_1 <- as.data.frame(Test_1[, -1])
  
  #### Step 2: Feature Selection with Boruta ####
  boruta <- Boruta(x = Train_genes_1, 
                   y = Train_target_1)
  boruta_features <- names(boruta$finalDecision[boruta$finalDecision %in% "Confirmed"])
  
  #### Save selected features ####
  write.csv(boruta_features, paste0(dataset_name, "_Selected_Features.csv"), row.names = FALSE)
  
  Trim_data_2 <- Train_1[, boruta_features]
  Trim_data_2 <- as.data.frame(cbind(Target = Train_1$Target, Trim_data_2))
  
  ##### Step 3: Second Splitting (80/20) ####
  index_2 <- caret::createDataPartition(Trim_data_2$Target, p = 0.8, list = FALSE)
  Train_2 <- Trim_data_2[index_2, ]
  Test_2 <- Trim_data_2[-index_2, ]
  
  Train_target_2 <- as.factor(Train_2$Target)
  Train_genes_2 <- as.data.frame(Train_2[, -1])
  Test_target_2 <- as.factor(Test_2$Target)
  Test_genes_2 <- as.data.frame(Test_2[, -1])
  
  #### Step 4:Random Fores tModel ####
  Control <- caret::trainControl(method = "cv", 
                                 number = 10, 
                                 verboseIter = TRUE)
  
  #### Model Training #####
  RF_model <- caret::train(x = Train_genes_2, 
                           y = Train_target_2,
                           method = "rf",
                           trControl = Control)
  
  ############## Test set 1 (external set) #####################
  
  #### Model Prediction ####
  Rf_pred_1 <- predict(RF_model, Test_genes_1)
  
  #### Confusion Matrix ####
  Rf_conf_1 <- confusionMatrix(Rf_pred_1, Test_target_1, 
                               mode = "everything", positive = "1")
  
  Rf_results_1 <- rbind(as.matrix(Rf_conf_1, what = "overall"),
                        as.matrix(Rf_conf_1, what = "classes"))
  
  #### Calculate AUC ####
  actual <- as.numeric(Test_target_1) - 1
  pred <- as.numeric(Rf_pred_1) - 1
  
  Rf_auc_1 <- auc(actual, pred)
  
  #### Append AUC to the model results ####
  Rf_results_1 <- rbind(Rf_results_1, 
                        "AUC" = c(Rf_auc_1, rep(NA, ncol(Rf_results_1) - 1)))
  
  ############## Test set 2 (internal set) #####################
  Rf_pred_2 <- predict(RF_model, Test_genes_2)
  
  Rf_conf_2 <- confusionMatrix(Rf_pred_2, Test_target_2, 
                               mode = "everything", positive = "1")
  
  Rf_results_2 <- rbind(as.matrix(Rf_conf_2, what = "overall"),
                        as.matrix(Rf_conf_2, what = "classes"))
  
  actual <- as.numeric(Test_target_2) - 1
  pred <- as.numeric(Rf_pred_2) - 1
  Rf_auc_2 <- auc(actual, pred)
  
  Rf_results_2 <- rbind(Rf_results_2,
                        "AUC" = c(Rf_auc_2, rep(NA, ncol(Rf_results_2) - 1)))
  
  #### Step 5: Artificial Neural Network ####
  grid <- expand.grid(size = seq(from = 1, to = 20, by = 10), 
                      decay = seq(from = 0.1, to = 0.5, by = 0.1))
  
  ANN_Model <- caret::train(x = Train_genes_2,
                            y = Train_target_2, 
                            method = "nnet", 
                            tuneGrid = grid, 
                            preProc = c("center", "scale", "nzv"),
                            trControl = Control)
  
  ############## Test set 1 (external set) #####################
  ANN_pred_1 <- predict(ANN_Model, Test_genes_1)
  
  ANN_conf_1 <- confusionMatrix(ANN_pred_1, Test_target_1, 
                                mode = "everything", positive = "1")
  
  ANN_results_1 <- rbind(as.matrix(ANN_conf_1, what = "overall"),
                         as.matrix(ANN_conf_1, what = "classes"))
  
  actual <- as.numeric(Test_target_1) - 1
  pred <- as.numeric(Rf_pred_1) - 1
  ANN_auc_1 <- auc(actual, pred)
  
  ANN_results_1 <- rbind(ANN_results_1, 
                         "AUC" = c(ANN_auc_1, rep(NA, ncol(ANN_results_1) - 1)))
  
  ############## Test set 2 (internal set) #####################
  ANN_pred_2 <- predict(ANN_Model, Test_genes_2)
  
  ANN_conf_2 <- confusionMatrix(ANN_pred_2, Test_target_2,
                                mode = "everything", positive = "1")
  
  ANN_results_2 <- rbind(as.matrix(ANN_conf_2, what = "overall"), 
                         as.matrix(ANN_conf_2, what = "classes"))
  
  actual <- as.numeric(Test_target_2) - 1
  pred <- as.numeric(Rf_pred_2) - 1
  ANN_auc_2 <- auc(actual, pred)
  
  ANN_results_2 <- rbind(ANN_results_2, 
                         "AUC" = c(ANN_auc_2, rep(NA, ncol(ANN_results_2) - 1)))
  
  
  #################### Combine RF and ANN results ########################
  combined_results <- cbind(Rf_results_1, Rf_results_2,
                            ANN_results_1, ANN_results_2)
  
  #### Assign column names ####
  colnames(combined_results) <- c("Rf test_1 results", 
                                  "Rf test_2 results",
                                  "ANN test_1 results", 
                                  "ANN test_2 results")
  
  # Save Results
  write.csv(Rf_results, paste0(dataset_name, "_results.csv"), row.names = TRUE)
}

  #### Apply pipeline to each dataset ####
  for (i in seq_along(datasets)) {
  apply_pipeline(datasets[[i]], dataset_names[i])
}
