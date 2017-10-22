############################ SVM Number Recognition #################################
# 1. Business Understanding
# 2. Objective
# 3. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

#####################################################################################

# 1. Business Understanding: 

#Consider an image of a digit submitted by a user via a scanner, a tablet, or other digital devices. 
#The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 

#####################################################################################

# 2. Objective

#Required to develop a model using Support Vector Machine which should correctly classify the handwritten digits 
#based on the pixel values given as features.

#####################################################################################

# 3. Data Understanding: 

# The Data contains features of each of the pixel of a particular Image of the digit i.e 784 features
# corresponding to each of the pixel on a 28 * 28 dimension image
# Two separate datasets are given each for Train and Test 
# Train Attributes
# Number of Instances:  60,000
# Number of Attributes: 784
# Test Attributes
# Number of Instances:  10,000
# Number of Attributes: 784


#3. Data Preparation: 

####################################################################################

#Loading Neccessary libraries

#install.packages("caret")
#install.packages("kernlab")
#install.packages("readr")
#install.packages("ggplot2")
#install.packages("cowplot")

#Loading Train & Test Data

numbers_train<-read.csv('mnist_train.csv',header = FALSE,stringsAsFactors = FALSE)

numbers_test<-read.csv('mnist_test.csv',header = FALSE,stringsAsFactors = FALSE)

#Understanding Dimensions

dim(numbers_train)

#Structure of the dataset

str(numbers_train)

#printing first few rows

head(numbers_train)

#Exploring the data

summary(numbers_train)

#Plotting the graph to find the spread of dependent variable in Train & Test Dataset

digit_spread<-plot_grid(ggplot(numbers_train,aes(numbers_train$V1))+geom_point(stat = "count")+
                        ggtitle("Digit Spread count")+labs(x = "Digit Number"),
                        ggplot(numbers_test,aes(numbers_test$V1))+geom_point(stat = "count")+
                        ggtitle("Digit Spread count")+labs(x = "Digit Number"),align="h")
digit_spread

#The spread of the data of all the classes in both the Train $ Test dataset seem to be even

#checking missing value

sum(is.na(numbers_train))
sum(is.na(numbers_test))

#There are no missing values in any of the testing and training datasets

#Data Sampling

#Extracting 25% of the original MNIST dataset as Sample

num_train<-numbers_train[sample(nrow(numbers_train),15000),]

num_test<-numbers_test[sample(nrow(numbers_test),2500),]

#Constructing Model

#Using Linear Kernel

Model_linear <- ksvm(V1~ ., data = num_train, scale = FALSE, kernel = "vanilladot")
Eval_linear  <- predict(Model_linear, num_test)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,num_test$V1)

############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 2 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

grid <- expand.grid(.C=c(seq(1,5,by=1)))

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.


#SVM on the train dataset
Sys.time()
fit.svm <- train(V1~., data=num_train, method="svmLinear", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)
Sys.time()

print(fit.svm)

plot(fit.svm)

######################################################################
# Checking overfitting - Linear - SVM
######################################################################

# Validating the model results on test data
evaluate_linear<- predict(fit.svm, num_test)
confusionMatrix(evaluate_linear, num_test$V1)

#The Accuracy for different Values of C ranging from 1 to 5 is equal i.e 90.7%
#Hence building the model to apply Non-Linear SVM

############ Building Non Linear SVM Using PCA Technique ##############
#The PCA technique is selected taking into consideration the number of dimensions in the given dataset is huge 
#equal to 784,this will contain lots of dimensions which are highly correlated,PCA technique will help in reducing
#the number of dimensions for the SVM model to be efficient.

#Making our target class to factor

num_train$V1<-factor(num_train$V1)
num_test$V1<-factor(num_test$V1)

#SVM Model application

X<-num_train[,-1]
Y<-num_train[,1]

trainLabel<-Y

#Scaling the Train dataset before PCA Analysis based on the standarad RGB pixel format range

Xreduced<-X/255

#Creating the Correlation dataframe to find the correlation between all the columns

Xcov<-cov(Xreduced)

#Applying the PCA function prcomp on the Correlated Train Dataframe
#The original dataframe has some columsn with zero variance hence scaling parameter cannot be applied in prcomp function
Xpca<-prcomp(Xcov)

#Check the derived parameters of PCA
names(Xpca)
summary(Xpca)

#Viewing the Rotation parameter as it shows the amount of variance captured by each of the Principal Component
Xpca$rotation

Xpca$rotation[1:5,1:4]

dim(Xpca$x)
biplot(Xpca,scale = 0)


#Check the variation of the first 10 components

#compute standard deviation of each principal component
std_dev <- Xpca$sdev

#compute variance
pr_var <- std_dev^2

#check variance of first 10 components
pr_var[1:10]

#Proportion of variance of each of the component

prop_varex <- pr_var/sum(pr_var)

#[1]  0.250315554 0.163733049 0.125994395 0.095348437 0.072994275 0.060038750
#[7]  0.033987012 0.027491924 0.023048584 0.018136010 0.014079377 0.012637837
#[13] 0.009882612 0.009350031 0.008149772 0.007009923 0.005765739 0.005117864
#[19] 0.004582152 0.004468402 0.003929167 0.003276196 0.003089501 0.002778813
#[25] 0.002524222 0.002369309 0.002152941 0.002095463 0.001805445 0.001545902

#It can be seen from the above reults that most of the variance is captured by the first ten Principal components

#Plotting the entire entire Prop_variance to find the number of Principal components to be included in the final model

plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

sum(prop_varex[1:37])

#It can be observed from the graph as well as from the above calculation that 98.5% variance is covered by the first 37
#Principal components

trainLabel<-as.factor(trainLabel)

# tranforming the dataset/ applying PCA to normalized-raw data

Xfinal<-as.matrix(Xreduced) %*% Xpca$rotation[,1:37]

model_svm<-svm(Xfinal, trainLabel, kernel="polynomial")

# predictions on the Test Data Sets

testLabel<-as.factor(num_test[,1])

# Applying PCA to test data,using the same Scaling conversion of RGB so as to avoid
# any unwanted bias and variance in the scales of Train and test dataset

testreduced<-num_test[,-1]/255

testfinal<-as.matrix(testreduced) %*% Xpca$rotation[,1:37]

# calculating accuracies
prediction<-predict(model_svm, testfinal, type="class")
confusionMatrix(prediction,num_test$V1)

#The Accuracy on the test Data is around 0.9672
#Statistics by Class:

#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
#Sensitivity            0.9872   0.9966   0.9412   0.9565   0.9573   0.9444   0.9719
#Specificity            0.9969   0.9973   0.9969   0.9952   0.9974   0.9978   0.9982
#Pos Pred Value         0.9706   0.9799   0.9717   0.9524   0.9739   0.9779   0.9837
#Neg Pred Value         0.9987   0.9995   0.9933   0.9956   0.9956   0.9943   0.9969
#Prevalence             0.0936   0.1172   0.1020   0.0920   0.0936   0.0936   0.0996
#Detection Rate         0.0924   0.1168   0.0960   0.0880   0.0896   0.0884   0.0968
#Detection Prevalence   0.0952   0.1192   0.0988   0.0924   0.0920   0.0904   0.0984
#Balanced Accuracy      0.9920   0.9969   0.9690   0.9758   0.9773   0.9711   0.9851

#Class: 7 Class: 8 Class: 9
#Sensitivity            0.9618   0.9791   0.9704
#Specificity            0.9969   0.9943   0.9928
#Pos Pred Value         0.9730   0.9474   0.9424
#Neg Pred Value         0.9955   0.9978   0.9964
#Prevalence             0.1048   0.0956   0.1080
#Detection Rate         0.1008   0.0936   0.1048
#Detection Prevalence   0.1036   0.0988   0.1112
#Balanced Accuracy      0.9794   0.9867   0.9816