library(readr)
library(dplyr)
library(stargazer)
library(caret)
library(pROC)
library(ggplot2)
library(gridExtra)

# setting up Train data
train_X <- read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/hill-valley/X.dat", 
                      col_names = FALSE)
x_mean=as.numeric(colMeans(train_X))
x_sd =as.numeric(apply(train_X,2,sd))
train_X1 <- cbind(rep(1,nrow(train_X)),scale(train_X,x_mean,x_sd))
dim(train_X1)
train_Y <- read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/hill-valley/Y.dat", 
                      col_names = FALSE)
data_train = list(x=as.matrix(train_X1),y=as.matrix(train_Y));str(data_train)

# setting up Test data
test_X <- read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/hill-valley/Xtest.dat", 
                     col_names = FALSE)
test_X1 <- cbind(rep(1,nrow(test_X)),scale(test_X,x_mean,x_sd))
dim(test_X1)
test_Y <- read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/Datasets/hill-valley/Ytest.dat", 
                     col_names = FALSE)
data_test = list(x=as.matrix(test_X1),y=as.matrix(test_Y));str(data_test)

# update Function

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


result=logistic(10000,data_train,data_test,eta=0.01)
result$Log.likelihood
result$ROC.plot
result$Table
