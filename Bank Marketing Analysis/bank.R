# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
# Step 1: Download the dataset
bank_data_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
dest_file <- "bank-additional.zip"
cat("Downloading dataset...\n")
download.file(bank_data_url, destfile = dest_file, mode = "wb")

# Step 2: Extract the dataset
cat("Extracting dataset...\n")
unzip(dest_file, exdir = "bank-data")

# Step 3: Load the dataset
csv_file <- "bank-data/bank-additional/bank-additional-full.csv"
cat("Loading dataset...\n")
data <- read.csv(csv_file, sep = ";")

# Step 4: Explore the dataset
cat("\n=== Dataset Exploration ===\n")

# Display the first few rows
cat("\nFirst few rows of the dataset:\n")
print(head(data))

# Check the structure of the dataset
cat("\nStructure of the dataset:\n")
print(str(data))

# Get a summary of the dataset
cat("\nSummary of the dataset:\n")
print(summary(data))

# Check for missing values
cat("\nNumber of missing values in the dataset:\n")
print(sum(is.na(data)))

# Clean up: Remove the downloaded ZIP file and extracted folder (optional)
cat("\nCleaning up...\n")
unlink(dest_file)  # Delete the ZIP file
unlink("bank-data", recursive = TRUE)  # Delete the extracted folder

cat("\nProcess complete!\n")
# Inspect the dataset
str(data)
summary(data)

# Step 2: Data Preprocessing
# Convert categorical variables to factors
data$job <- as.factor(data$job)
data$marital <- as.factor(data$marital)
data$education <- as.factor(data$education)
data$default <- as.factor(data$default)
data$housing <- as.factor(data$housing)
data$loan <- as.factor(data$loan)
data$contact <- as.factor(data$contact)
data$month <- as.factor(data$month)
data$poutcome <- as.factor(data$poutcome)
data$y <- as.factor(data$y)

# Check for missing values
sum(is.na(data))

# Step 3: Exploratory Data Analysis (EDA)
# Univariate analysis
ggplot(data, aes(x = age)) + geom_histogram(binwidth = 5, fill = "blue")
ggplot(data, aes(x = balance)) + geom_histogram(binwidth = 1000, fill = "green")

# Bivariate analysis
ggplot(data, aes(x = job, fill = y)) + geom_bar(position = "fill")
ggplot(data, aes(x = education, fill = y)) + geom_bar(position = "fill")

# Step 4: Feature Engineering
# Create age groups
data <- data %>%
  mutate(age_group = cut(age, breaks = c(0, 30, 40, 50, 60, 100), labels = c("<30", "30-40", "40-50", "50-60", "60+")))

# Step 5: Model Building
# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$y, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Train a Random Forest model
model <- randomForest(y ~ ., data = trainData, ntree = 100, importance = TRUE)

# Step 6: Model Evaluation
# Make predictions on the test set
predictions <- predict(model, testData)

# Confusion matrix
confusionMatrix(predictions, testData$y)

# ROC curve and AUC
roc_curve <- roc(testData$y, as.numeric(predictions))
plot(roc_curve)
auc(roc_curve)

# Step 7: Deployment (Optional)
# Save the model
saveRDS(model, "term_deposit_model.rds")

# Load the model for future use
# loaded_model <- readRDS("term_deposit_model.rds")

