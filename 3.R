library(readr)
library(readsparse)
library(readr)
library(dplyr)
library(stargazer)
library(caret)
library(pROC)
library(ggplot2)
library(gridExtra)

#to read sparse data

train <-read.sparse("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/dexter/dexter_train.data")

train_y <- as.matrix(read_csv("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/dexter/dexter_train.labels", 
                                   col_names = FALSE))
test <-read.sparse("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/dexter/dexter_valid.data")
test_y <- as.matrix(read_csv("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/dexter/dexter_valid.labels", 
                                   col_names = FALSE))

# train mean and sd
x_mean=as.numeric(colMeans(as.matrix(train$X)))
x_sd =as.numeric(apply(as.matrix(train$X),2,sd))

# Rename labels
train_y[train_y== -1] = 0
test_y[test_y== -1] = 0
#setup data
train$x <- as.matrix(cbind(rep(1,nrow(train$X)),scale(train$X,x_mean,x_sd)))
test$x <- as.matrix(cbind(rep(1,nrow(test$X)),scale(test$X,x_mean,x_sd)))

X = rbind(train$x, test$x)
X = X[, colSums(is.na(X)) == 0]
data_train = list(y=train_y,x=as.matrix(X[1:300,]));str(data_train)
data_test= list(y=test_y,x=as.matrix(X[301:600,]));str(data_test)




logistic = function(n_iter=300,
                    data_train,
                    data_test,
                    eta=0.0001,
                    lambda = 0.0001,
                    w_init = as.matrix(rep(0, ncol(data_train$x))))
{
  w_new = w_init
  n = 1
  my.list = list()
  log_likelihood = array()
  iteration = array()
  # Updating w
  while (n <= n_iter)
  {
    w = w_new
    #calculating log likelihood
    pred = data_train$x %*% w
    lik = data_train$y * pred - log(1 + exp(pred))
    log_likelihood[n] = sum(as.numeric(lik))
    
    #Calculating Gradient
    v = as.matrix(data_train$y - (exp(pred) / 1 + exp(pred)))
    grad = t(data_train$x) %*% v
    #updating coefficients
    w_new = w - eta * lambda * w + (eta / nrow(data_train$y)) * grad
    #keeping iteration number
    iteration[n] = n
    #updating iteration
    n = n + 1
  }  
  
  #plotting log likelihood
  data=data.frame(iteration,log_likelihood)
  my.list$Log.likelihood = ggplot(data, aes(x = iteration)) + geom_line(aes(y = log_likelihood))
  
  #train data
  link_train = data_train$x %*% w_new
  prob_train = exp(link_train)/(1+exp(link_train))
  y_train = data_train$y
  roc_train = roc(as.numeric(y_train), as.numeric(prob_train))
  threshold_train = as.numeric(coords(roc_train, "best", ret = "threshold"))
  y_hat_train = as.factor(ifelse(prob_train > threshold_train, 1, 0))
  levels(y_hat_train) = c("0", "1")
  cm_train = confusionMatrix(as.factor(y_train), as.factor(y_hat_train))
  train_miss = as.numeric(1 - cm_train$byClass['Balanced Accuracy'])
  # test_data
  link_test = data_test$x %*% w_new
  y_test = data_test$y
  prob_test= exp(link_test)/(1+exp(link_test))
  roc_test = roc(as.numeric(y_test), as.numeric(prob_test))
  threshold_test = as.numeric(coords(roc_test, "best", ret = "threshold"))
  y_hat_test = as.factor(ifelse(prob_test > threshold_test, 1, 0))
  levels(y_hat_test) = c("0", "1")
  cm_test = confusionMatrix(as.factor(y_test), as.factor(y_test))
  test_miss = as.numeric(1 - cm_test$byClass['Balanced Accuracy'])
  # Thresholds and Misclassification 
  my.list$Table = matrix(c(train_miss,threshold_train,test_miss,threshold_test),nrow=2,byrow=TRUE)
  rownames(my.list$Table) = c("Train","Test")
  colnames(my.list$Table) = c("Misclassification.Prob","Threshold")
  # Roc plot
  my.list$ROC.plot = ggroc(list(Train = roc_train, Test = roc_test ))+geom_abline(slope=1,intercept = 1,color = "blue") 
  return(my.list)
}
r=logistic(1000,data_train,data_test,eta=0.0001)
r




