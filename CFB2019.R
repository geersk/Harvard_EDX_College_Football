# EDX9 Capstone: Predicting a Win-Loss Record in American College Football

# Libraries
library(tidyverse)
library(data.table)
library(dplyr)
library(ggplot2)
library(Hmisc)
library(ggpubr)
library(corrplot)
library(nnet)
library(ROCR)
library(caret)
library(pROC)
library(mlbench)
library(randomForest)

# Install packages if necessary
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(Hmisc)) install.packages("Hmisc", repos = "http://cran.us.r-project.org")
if(!require(ggpubr)) install.packages("ggpubr", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(ROCR)) install.packages("ROCR", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(mlbench)) install.packages("mlbench", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

# Data Source
FB_Analysis <- read.csv("CFB2019.txt")

# Data Dimensions
dim(FB_Analysis)

# Data Structure
str(FB_Analysis [1:15])

# First Rows of Data
head(FB_Analysis [1:8])

# Visualization of Team Wins
ggplot(data = FB_Analysis) + 
  geom_bar(aes(x = Wins), col = "gold3") +
  labs(title = "2019 College Football Season", x = "Wins", y = "Teams") +
  theme(title = element_text(color = "Purple3", face = "bold", size = 14))

# Performance Metrics
ggplot(data = FB_Analysis) +
  geom_point(aes(x = Off.Rank, y = Def.Rank, alpha = Wins, size = Wins)) +
  labs(title = "Performance Metrics", x = "Offense Rank", y = "Defense Rank") +
  theme(title = element_text(color = "Purple3", face = "bold", size = 14))

# Ranked Observations
Ranked_Observations <- FB_Analysis %>% select(matches("Rank|Wins"))

# Here are the first 8 performance metrics (starting with the second column):
head(Ranked_Observations [1:9])

# Logistic Regression Model
# We split the data into a 'train' set (80%) and a 'test' set (20%).
set.seed(111)
ind <- sample(2, nrow(Ranked_Observations), replace = TRUE, prob = c(0.8, 0.2))
train <- Ranked_Observations[ind==1,]
test <- Ranked_Observations[ind==2,]

# Here you can see the first six rows of each data set.
head(train [1:3])
head(test [1:3])

# This code runs our LR model, with the wins predicted from all 24 Ranked_Observations.
LRM_model <- lm(Wins ~ ., data = train)
summary(LRM_model)

# To improve our LR model (for "LRM_model2"), we choose to leverage all 7 starred P values from above.
LRM_model2 <- lm(Wins ~ First.Down.Def.Rank + X4rd.Down.Def.Rank + Pass.Def.Rank + Rushing.Def.Rank + Scoring.Def.Rank + Scoring.Off.Rank + Time.of.Possession.Rank, data = train)
summary(LRM_model2)

# Next, we use our new model to guess the number of wins each team will have in the train data:
LRM_predictions_train <- predict(LRM_model2, train, type = 'response')
head(LRM_predictions_train [1:7])

# Confusion Matrix: LR train
LRM_Conf_train <- ifelse(round(LRM_predictions_train) == train$Wins, 1, 0)
LRM_Conf_Tab_train <- table(Predicted = LRM_Conf_train, Actual = train$Wins)
LRM_Conf_Tab_train

# Here is the percentage of wins LR correctly predicted in the train data. 
options(digits = 5)
sum(LRM_Conf_Tab_train[2,])/sum(LRM_Conf_Tab_train)
                                                           
# Next, let's try our LR model against the test data:
LRM_predictions_test <- predict(LRM_model2, test, type = 'response')
head(LRM_predictions_test [1:7])

# This Confusion Matrix shows the results:
LRM_Conf_test <- ifelse(round(LRM_predictions_test) == test$Wins, 1, 0)
LRM_Conf_Tab_test <- table(Predicted = LRM_Conf_test, Actual = test$Wins)
LRM_Conf_Tab_test

# Here is our percentage score against the test data:
sum(LRM_Conf_Tab_test[2,])/sum(LRM_Conf_Tab_test)

# K-Nearest Neighbors Model
# We first split our data set into a 'train' set (80%) and a 'test' set (20%):
set.seed(222)
ind2 <- sample(2, nrow(Ranked_Observations), replace = TRUE, prob = c(0.8, 0.2))
train2 <- Ranked_Observations[ind2==1,]
test2 <- Ranked_Observations[ind2==2,]

# The test data is randomly chosen from the complete data set.
head(train2 [1:3])
head(test2 [1:3])

# Here is the code to build our KNN model.
trControl <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 3)
set.seed(2222)
KNN_model <- train(Wins ~ .,
                   data = train2,
                   method = 'knn',
                   tuneLength = 20,
                   trControl = trControl,
                   preProc = c("center", "scale"))
KNN_model

# Here is a visualization of the RMSE selection process.
plot(KNN_model, col = "Purple3")

# You can see that the RMSEs in both the train and test sets are quite similar.
KNN_predict_train <- predict(KNN_model, newdata = train2)
KNN_predict_test <- predict(KNN_model, newdata = test2)
RMSE(KNN_predict_train, train2$Wins)
RMSE(KNN_predict_test, test2$Wins)

# These two charts show the similarity of the KNN predictive capability in both the train and test sets.
plot(KNN_predict_train ~ train2$Wins, col = "Purple3")
plot(KNN_predict_test ~ test2$Wins, col = "Purple3")

# In the table below, KNN calculates variable importance, sorting the top 20 in descending order.
varImp(KNN_model)

# Now let's see how well KNN is able to predict a team's win total for the season.
KNN_CM_train <- predict(KNN_model, train2, type = 'raw')
head(KNN_CM_train [1:9])

# Here is the KNN Confusion Matrix for the train data:
KNN_CM_train2 <- ifelse(round(KNN_CM_train) == train2$Wins, 1, 0)
KNN_CM_train3 <- table(Predicted = KNN_CM_train2, Actual = train2$Wins)
KNN_CM_train3

# How well did it work?
options(digits = 5)
sum(KNN_CM_train3[2,])/sum(KNN_CM_train3)

# Now let's run KNN against the test data.
KNN_CM_test <- predict(KNN_model, test2, type = 'raw')
KNN_CM_test

# Here is the KNN Confusion Matrix for the test data:
KNN_CM_test2 <- ifelse(round(KNN_CM_test) == test2$Wins, 1, 0)
KNN_CM_test3 <- table(Predicted = KNN_CM_test2, Actual = test2$Wins)
KNN_CM_test3

# And the KNN percentage for the test data.
options(digits = 5)
sum(KNN_CM_test3[2,])/sum(KNN_CM_test3)

# Random Forest Model
# First we split the data into a 'train' set (80%) and a 'test' set (20%).
set.seed(333)
ind3 <- sample(2, nrow(Ranked_Observations), replace = TRUE, prob = c(0.8, 0.2))
train3 <- Ranked_Observations[ind3==1,]
test3 <- Ranked_Observations[ind3==2,]

# The split data sets are shown below.
head(train3 [1:3])
head(test3 [1:3])

# This code creates our RF model:
rf <- randomForest(Wins ~ ., data = train3)
print(rf)

# The model's error rate can be seen in the plot below.
plot(rf, main = "RF: Error Rate", col = "Purple3")

# Here is a histogram of the treesize, or number of nodes.
hist(treesize(rf), main = "RF: Treesize", col = "Purple3")

# Here are the model's list of attributes.
attributes(rf)

# Let's run the importance function, which provides a table of node "purity" within the model.
importance(rf)

# Let's visualize RF's choices in a dotchart.
varImpPlot(rf, n.var=12, main = "Variable Importance", col = "Purple3")

# Now it is time to see how well RF's hybrid approach can predict the number of a team's wins.
RF_CM_train <- predict(rf, train3, type = 'response')
head(RF_CM_train [1:9])

# Here is the Confusion Matrix for the RF train set:
RF_CM_train2 <- ifelse(round(RF_CM_train) == train3$Wins, 1, 0)
RF_CM_train3 <- table(Predicted = RF_CM_train2, Actual = train3$Wins)
RF_CM_train3

# And the percentage of correct predictions for the train set:
options(digits = 5)
sum(RF_CM_train3[2,])/sum(RF_CM_train3)

# Let's see whether this success can be duplicated in the test set.
RF_CM_test <- predict(rf, test3, type = 'response')
head(RF_CM_test [1:9])

# Here is the Confusion Matrix for the RF test set:
RF_CM_test2 <- ifelse(round(RF_CM_test) == test3$Wins, 1, 0)
RF_CM_test3 <- table(Predicted = RF_CM_test2, Actual = test3$Wins)
RF_CM_test3

# And our final percentage:
options(digits = 5)
sum(RF_CM_test3[2,])/sum(RF_CM_test3)

# To gather a bit more data on this, let's check the RMSE of both the train set and test set.
RF_predict_train <- predict(rf, newdata = train3)
RF_predict_test <- predict(rf, newdata = test3)
RMSE(RF_predict_train, train3$Wins)
RMSE(RF_predict_test, test3$Wins)

# We can plot the RMSE disparity thus:
plot(RF_predict_train ~ train3$Wins, col = "Purple3")
plot(RF_predict_test ~ test3$Wins, col = "Purple3")
