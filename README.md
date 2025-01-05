# **ML-Based Classification of IUGR and Preeclampsia**
Poster Presented at "1st International Conference on Maternal Fetal Medicine"
Organized By Aga Khan University, Department of Obstetrics & Gynaecology Section of Maternal Fetal Medicine
on Saturday, November 25, 2023

This project employed a machine learning technique to analyze gene expression data for the prediction intrauterine growth restriction (IUGR) and pre-eclampsia (PE). 

The repository contains code for reproducibility of results
Here’s a detailed and descriptive GitHub `README.md` for your project, tailored to the workflow and codes provided:  

---

# **Machine Learning-Based Feature Selection for Gene Expression Data**  
  

## **Overview**  
This project focuses on leveraging machine learning techniques to classify gene expression data associated with Intrauterine Growth Restriction (IUGR) and Preeclampsia. The preprocessed(normalized) dataset used in this project was obtained from the NCBI GEO repository with accession number **[GSE114691](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE114691)**.  

The analysis pipeline involves:  
1. Preprocessing and transforming gene expression data.  
2. Applying feature selection using the **Boruta** algorithm.  
3. Training machine learning models, including **Random Forest (RF)** and **Artificial Neural Networks (ANN)**.  
4. Evaluating the models on internal and external test datasets using metrics such as **accuracy**, **sensitivity**, **specificity**, **F1-score**, and **AUC**.

## **Project Workflow**  

![image](https://github.com/user-attachments/assets/5fe3042e-bf18-4ba8-a6c1-c027ee82bfa9)  

---
### Data Preparation**  
1. Download the dataset from the given [link](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE114691)
2. Preprocess the gene expression data(normalized expression) by transposing the dataset and defining target groups:  
   - **Control**: Normal samples.  
   - **PE**: Preeclampsia samples.  
   - **PEIUGR**: Preeclampsia with IUGR samples.  
   - **IUGR**: IUGR samples.

# **Prerequisites**  
Install the following R libraries before running the scripts:  
```
install.packages(c("caret", "data.table", "dplyr", "tidyverse", "ggplot2", "Boruta", "pROC", "randomForest"))
```
# Load required libraries:   
```
library(caret)
library(data.table)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(Boruta)
library(pROC)
library(purrr)
library(randomForest)
```
# Import and transform Dataset:  
```
#Path to the .tsv.gz file 
file_path <- "path/to/your/file.tsv.gz" ##### Copy/paste file path here

# Read the compressed TSV file 
data <- read.delim(gzfile(file_path), sep = "\t")

data_t <- data.table::transpose(data,
                                keep.names = "RefID", 
                                make.names = "GeneID")
``` 
3. Create the following subsets for classification tasks:  
   - `subset_1`: Control vs PE.  
   - `subset_2`: Control vs PEIUGR.  
   - `subset_3`: Control vs IUGR.  
   - `subset_4`: PE vs IUGR.  

# Subsets creation:  
```
# Define group vector 
target_column <- c(
  rep("Control", 21),   # 21 Control samples
  rep("PE", 20),        # 20 PE samples
  rep("PEIUGR", 20),    # 20 PEIUGR samples
  rep("IUGR", 18)       # 18 IUGR samples
)
data_t <- cbind(Target = target_column, data_t[, -1])

# Creating subsets
subset_1 <- data_t[1:41,]
subset_2 <- data_t[c(1:21,42:61),]
subset_3 <- data_t[c(1:21,62:79),]
subset_4 <- data_t[c(22:41,62:79),]

# Convert target column to numeric
subset_1$Target <- ifelse(subset_1$Target == "Control", 0, 1)
subset_2$Target <- ifelse(subset_2$Target == "Control", 0, 1)
subset_3$Target <- ifelse(subset_3$Target == "Control", 0, 1)
subset_4$Target <- ifelse(subset_4$Target == "IUGR", 0, 1)
```

 # First Splitting 70/30 Externat Train and Test Sets:    
 ```
  index_1 <- caret::createDataPartition(subset_1$Target, p = 0.7, list = FALSE) replace subsets accordingly
  Train_1 <- subset_1[index_1, ]
  Test_1 <- subset_1[-index_1, ]
  
  Train_target_1 <- as.factor(Train_1$Target)
  Test_target_1 <- as.factor(Test_1$Target)
  Train_genes_1 <- as.data.frame(Train_1[, -1])
  Test_genes_1 <- as.data.frame(Test_1[, -1])

```  

### **Feature Selection Using Boruta**  
The **Boruta** algorithm identifies the most important genes contributing to classification.  
Selected features for each subset are saved as `.csv` files for further use.  

# Feature Selection with Boruta:  
```
boruta <- Boruta(x = Train_genes_1, y = Train_target_1)
boruta_features <- names(boruta$finalDecision[boruta$finalDecision %in% "Confirmed"])
write.csv(boruta_features, file = "Subset_1_Selected_Features.csv", row.names = FALSE)
```
# Second Splitting (80/20) Internal Train and Test Set
```
  Trim_data_2 <- Train_1[, boruta_features]
  Trim_data_2 <- as.data.frame(cbind(Target = Train_1$Target, Trim_data_2))

  index_2 <- caret::createDataPartition(Trim_data_2$Target, p = 0.8, list = FALSE)
  Train_2 <- Trim_data_2[index_2, ]
  Test_2 <- Trim_data_2[-index_2, ]
  
  Train_target_2 <- as.factor(Train_2$Target)
  Train_genes_2 <- as.data.frame(Train_2[, -1])
  Test_target_2 <- as.factor(Test_2$Target)
  Test_genes_2 <- as.data.frame(Test_2[, -1])
```

### **Model Training and Evaluation**  
# **With Random Forest (RF)**:  
```
Control <- trainControl(method = "cv",  
                        number = 10,  
                        verboseIter = TRUE)  
RF_model <- train(x = Train_genes_2,  
                  y = Train_target_2,  
                  method = "rf",
                  trControl = Control)  
```

# **With Artificial Neural Networks (ANN)**:  
```
grid <- expand.grid(size = seq(from = 1, to = 20, by = 10),
                    decay = seq(from = 0.1, to = 0.5, by = 0.1))
ANN_Model <- train(x = Train_genes_2,
                   y = Train_target_2,
                   method = "nnet",
                   tuneGrid = grid,
                   preProc = c("center", "scale", "nzv"),
                   trControl = Control)
```

### **Performance Metrics**  
Both models are evaluated using:  
- **Confusion Matrix** for accuracy, sensitivity, specificity, and F1-score.  
- **AUC** for measuring the area under the curve.  

# Random Forest Model evaluation:  
```
# External Test Set
Rf_pred_1 <- predict(RF_model, Test_genes_1)
Rf_conf_1 <- confusionMatrix(Rf_pred_1, Test_target_1, mode = "everything", positive = "1")
Rf_auc_1 <- auc(as.numeric(Test_target_1) - 1, as.numeric(Rf_pred_1) - 1)

# Internal Test Set
Rf_pred_2 <- predict(RF_model, Test_genes_2)
Rf_conf_2 <- confusionMatrix(Rf_pred_2, Test_target_2, mode = "everything", positive = "1")
Rf_auc_2 <- auc(as.numeric(Test_target_2) - 1, as.numeric(Rf_pred_2) - 1)
```
# Artificial Neural Network Model evaluation:  
```
# External Test Set
ANN_pred_1 <- predict(ANN_model, Test_genes_1)
ANN_conf_1 <- confusionMatrix(ANN_pred_1, Test_target_1, mode = "everything", positive = "1")
ANN_auc_1 <- auc(as.numeric(Test_target_1) - 1, as.numeric(ANN_pred_1) - 1)

# Internal Test Set
ANN_pred_2 <- predict(ANN_model, Test_genes_2)
ANN_conf_2 <- confusionMatrix(ANN_pred_2, Test_target_2, mode = "everything", positive = "1")
ANN_auc_2 <- auc(as.numeric(Test_target_2) - 1, as.numeric(ANN_pred_2) - 1)
```

### **Step 5: Combining Results**  
Results from both RF and ANN models are combined into a single dataframe and saved as a `.csv` file.  

# Saving results:  
```
# RF Results 
Rf_results_1 <- rbind(as.matrix(Rf_conf_1, what = "overall"),
                         as.matrix(Rf_conf_1, what = "classes"))

Rf_results_1 <- rbind(Rf_results_1, 
                         "AUC" = c(Rf_auc_1, rep(NA, ncol(Rf_results_1) - 1)))

Rf_results_2 <- rbind(as.matrix(Rf_conf_2, what = "overall"),
                         as.matrix(Rf_conf_2, what = "classes"))

Rf_results_2 <- rbind(Rf_results_2, 
                         "AUC" = c(Rf_auc_2, rep(NA, ncol(Rf_results_2) - 1)))


# ANN Results
ANN_results_1 <- rbind(as.matrix(ANN_conf_1, what = "overall"),
                         as.matrix(ANN_conf_1, what = "classes"))

ANN_results_1 <- rbind(ANN_results_1, 
                         "AUC" = c(ANN_auc_1, rep(NA, ncol(ANN_results_1) - 1)))

ANN_results_2 <- rbind(as.matrix(ANN_conf_2, what = "overall"),
                         as.matrix(ANN_conf_2, what = "classes"))

ANN_results_2 <- rbind(ANN_results_2, 
                         "AUC" = c(ANN_auc_2, rep(NA, ncol(ANN_results_2) - 1)))

# Combined results in one dataframe
combined_results <- cbind(Rf_results_1, Rf_results_2, ANN_results_1, ANN_results_2)
colnames(combined_results) <- c("Rf test_1 results", "Rf test_2 results", "ANN test_1 results", "ANN test_2 results")
write.csv(combined_results, file = "_Model_Results_IUGR-PE.csv"), row.names = TRUE)
```
### The workflow described above is repeated for all subsets e.g Subset_2, Subset_3 & Subset_4.
To avoid repeatition To avoid manually running the pipeline for each subset, you can refer to the integrated R script provided in this repository. The script automates the entire process across all subsets. Locate the integrated R script in the repository [IUGR_PE machine learning analysis.R](https://github.com/zehrhiz/ML-Based-Prediction-IUGR-PE/blob/fa26c9bfecf14b3d23d4d27fd781605883febb61/IUGR_PE%20machine%20learning%20analysis.R)
