#Customer churn modeling for telecommunication industry customers
  
install.packages("C50")
library(C50)
data(churn)
str(churnTrain)

churnTrain <- churnTrain[ , !names(churnTrain) %in% c("state", "area_code", "account_length")]

#2a splitting the data in to train/test - method 1
set.seed(2)
ind <- sample(2, nrow(churnTrain), replace = TRUE, prob = c(0.7, 0.3))
trainset <- churnTrain[ind == 1, ]
testset <- churnTrain[ind == 2, ]


#3 building classification model with recursive partitioning tree

library(rpart)

churn.rp <- rpart(churn ~ ., data = trainset)

#3a to retrieve the node detail of the classification tree
churn.rp

#3b examine the complexity parameter
printcp(churn.rp)

#3c plot the cost complexity parameter ..... NOTE
plotcp(churn.rp)

#3d examine the built model
summary(churn.rp)

#3e visualise the recursive partitioning tree
plot(churn.rp, margin = 0.1)
text(churn.rp, all = TRUE, ue.n = TRUE)

#3f Adjust the layout with parameters such as Uniform, Branch, Margin

plot(churn.rp, uniform = TRUE, branch=0.6, margin = 0.1)
text(churn.rp, all = TRUE, use.n = TRUE)

#3g measure the predictive performance of the recursive partitioning tree

predictions <- predict(churn.rp, testset, type = "class")

table(testset$churn, predictions)

#3h generate confusion matrix

library(lattice)
library(ggplot2)
library(caret)
confusionMatrix(table(predictions, testset$churn))


#3i pruning a recursive partitioning tree

#3i - 1 find minimum cross-validation error pf the classification tree model
min(churn.rp$cptable[ , "xerror"])

#3i - 2 locate the error with minimum cross-validation errors
which.min(churn.rp$cptable[ , "xerror"])

#3i - 3 get the cost complexity parameter of the record with minimum cross validation
churn.cp <- churn.rp$cptable[7, "CP"]
churn.cp

#3i - 4 prune the tree by setting tree parameter to the cp value of the record with minimum cross validation error
prune.tree <- prune(churn.rp, cp= churn.cp)

#3i - 5 visualise the classification tree by using the plot and text function
plot(prune.tree, margin = 0.1)
text(prune.tree, all = TRUE, use.n = TRUE)

#3i - 6 generate a classification tbale based on the pruned tree
predictions <- predict(prune.tree, testset, type="class")
table(testset$churn, predictions)

#3i - 7 generate a confusion matrix
confusionMatrix(table(predictions, testset$churn))


# Building classification model with a conditional inference tree

library(zoo)
library(sandwich)
library(party)

ctree.model <- ctree(churn ~ ., data = trainset)

ctree.model

help("binaryTree-class")


#visualising the conditional intference tree

plot(ctree.model)

daycharge.model <- ctree(churn ~ total_day_charge, data = trainset)
plot(daycharge.model)

# Measuring the prediction performance of a conditional tree

ctree.predict = predict(ctree.model, testset)
table(ctree.predict, testset$churn)
confusionMatrix(table(ctree.predict, testset$churn))

tr <- treeresponse(ctree.model, newdata = testset[1:5, ])
tr


# Classifying data with K-nearest neighbor classifier

install.packages("class")
library(class)

# replace "yes" and "no" labels with 0 and 1 

levels(trainset$international_plan) <- list("0" = "no", "1" = "yes")
levels(trainset$voice_mail_plan) <- list("0" = "no", "1" = "yes")
levels(testset$international_plan) <- list("0" = "no", "1" = "yes")
levels(testset$voice_mail_plan) <- list("0" = "no", "1" = "yes")

# use k - nearest neighbour classification method on the training dataset and test dataset 

churn.knn <- knn(trainset[ , ! names(trainset) %in% c("churn") ], testset [ , ! names(testset) %in% c("churn") ], trainset$churn, k =3 )

summary(churn.knn)

table(testset$churn, churn.knn)

confusionMatrix(table(testset$churn, churn.knn))

# classifying data with logistic regression

fit <- glm(churn ~., data = trainset, family = binomial)

summary(fit)

# reduce insignificant variables to avoid missclassification. use ONLY significant variables ERROR !!!!

fit <- glm(churn ~ international_plan + voice_mail_plan + total_intl_calls + number_customer_service_calls, data = trainset, family = binomial)
summary(fit)

pred <- predict(fit, testset, type = "response")
class <- pred > .5
summary(class)

tb <- table(testset$churn, class)

tb

# convert the statistics into classification table and generate confusion mtrix

churn.mod <- ifelse(testset$churn == "yes", 1, 0)
pred_class <- churn.mod

pred_class[pred<=.5] <- 1 - pred_class[pred<=.5]

ctb <- table(churn.mod, pred_class)
ctb

confusionMatrix(ctb)

# classifying data with Naive bayes

library(e1071)

classifier <- naiveBayes(trainset[, !names(trainset) %in% c("churn")], trainset$churn)

classifier


#generate classification table
bayes.table <- table(predict(classifier, testset[ , !names(testset) %in% c("churn")]), testset$churn)

bayes.table

confusionMatrix(bayes.table)


#classifying data with a support vector machine

library(e1071)

model <- svm(churn~., data = trainset, kernel = "radial", cost = 1, gamma = 1/ncol(trainset))
summary(model)


# Predicting labels based on a model trained by Support Vector Machine

svm.pred <- predict(model, testset[, !names(testset) %in% c("churn")])
svm.table <- table(svm.pred, testset$churn)
svm.table

classAgreement(svm.table)

library(caret)
confusionMatrix(svm.table)

# Tuning a support vector machine
tuned <- tune.svm(churn ~ ., data = trainset, gamma = 10^(-6:-1), cost = 10^(1:2))
summary(tuned)
model.tuned <- svm(churn ~ ., data = trainset, gamma = tuned$best.parameters$gamma, cost = tuned$best.parameters$cost)

summary(model.tuned)
svm.tuned.pred <- predict(model.tuned, testset[, !names(testset) %in% c("churn")])
svm.tuned.table <- table(svm.tuned.pred, testset$churn)
svm.tuned.table

classAgreement(svm.tuned.table)
confusionMatrix(svm.tuned.table)

# Measuring Prediction Performance using ROCR

install.packages("ROCR")
library(ROCR)

#train the svm model
svmfit <- svm(churn ~., data = trainset, prob = TRUE)

#make predictions based on trained model on the test dataset with probability set as true
pred <- predict(svmfit, testset[, !names(testset) %in% c("churn")], probability = TRUE)

#obtain the probability of labels with "yes"
pred.prob <- attr(pred, "probabilities")
pred.to.roc <- pred.prob[, 2]

#use the prediction function to generate the prediction result
pred.rocr <- prediction(pred.to.roc, testset$churn)

# use the perfromance function to obtain the performance measurement
perf.rocr <- performance(pred.rocr, measure = "auc", x.measure = "cutoff")
perf.tpr.rocr <- performance(pred.rocr, "tpr", "fpr")

#visualize the ROC curve using the plot function
plot(perf.tpr.rocr, colorize = T, main = paste("AUC:", (perf.rocr@y.values)))

#Comparing an ROC curve using "caret" package

install.packages("pROC")
library(pROC)

#setting up a 10 fold cross validation in 3 repititions
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE, summaryFunction = twoClassSummary)

#train classifier using glm
glm.model <- train(churn ~ ., data = trainset, method = "glm", metric = "ROC", trControl = control)

#train classifier using svm
library(kernlab)
svm.model <- train(churn ~ ., data = trainset, method = "svmRadial", metric = "ROC", trControl = control)

#train classifier using rpart// doesnt work unless you replace "yes" and "no" labels with 0 and 1 
rpart.model <- train(churn ~ ., data = trainset, method = "rpart", metric = "ROC", trControl = control)



#now make predictions separately based on different trained models

glm.probs <- predict(glm.model, testset[, ! names(testset) %in% c("churn")], type = "prob")
svm.probs <- predict(svm.model, testset[, ! names(testset) %in% c("churn")], type = "prob")
rpart.probs <- predict(rpart.model, testset[, ! names(testset) %in% c("churn")], type = "prob")


# generate ROC curve for each model in one plot

rpart.ROC <- roc(response = testset[, c("churn")], predictor = rpart.probs$yes, levels = levels(testset[, c("churn")]))
rpart.ROC
plot(rpart.ROC, add = TRUE, col = "blue")


levels(trainset$international_plan) <- list("no" = "0", "yes" = "1" )
levels(trainset$voice_mail_plan) <- list("no" = "0", "yes" = "1" )
levels(testset$international_plan) <- list("no" = "0", "yes" = "1" )
levels(testset$voice_mail_plan) <- list("no" = "0", "yes" = "1" )

glm.ROC <- roc(response = testset[, c("churn")], predictor = glm.probs$yes, levels = levels(testset[, c("churn")]))
glm.ROC
plot(glm.ROC, type = "S", col = "red")

svm.ROC <- roc(response = testset[, c("churn")], predictor = svm.probs$yes, levels = levels(testset[, c("churn")]))
svm.ROC
plot(svm.ROC, add = TRUE, col = "green")
