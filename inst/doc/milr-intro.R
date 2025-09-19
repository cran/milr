## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(cache = 2, cache.lazy = FALSE, tidy = FALSE, warning = FALSE)

## ----eval=FALSE---------------------------------------------------------------
# milr(y, x, bag, lambda, numLambda, lambdaCriterion, nfold, maxit)
# softmax(y, x, bag, alpha, ...)

## ----eval=FALSE---------------------------------------------------------------
# fitted(object, type)
# predict(object, newdata, bag_newdata, type)

## ----DGP1---------------------------------------------------------------------
library(milr)
library(pipeR)
set.seed(99)
# set the size of dataset
numOfBag <- 30
numOfInstsInBag <- 3
# set true coefficients: beta_0, beta_1, beta_2, beta_3
trueCoefs <- c(-2, -1, 2, 0.5)
trainData <- DGP(numOfBag, numOfInstsInBag, trueCoefs)
colnames(trainData$X) <- paste0("X", 1:ncol(trainData$X))
(instanceResponse <- as.numeric(with(trainData, tapply(Z, ID, any))))

## ----EST2---------------------------------------------------------------------
# fit milr model
milrFit_EST <- milr(trainData$Z, trainData$X, trainData$ID, lambda = 1e-7)
# call the Wald test result
summary(milrFit_EST)
# call the regression coefficients
coef(milrFit_EST)

## ----EST----------------------------------------------------------------------
fitted(milrFit_EST, type = "bag") 
# fitted(milrFit_EST, type = "instance") # instance-level fitted labels
table(DATA = instanceResponse, FITTED = fitted(milrFit_EST, type = "bag")) 
# predict for testing data
testData <- DGP(numOfBag, numOfInstsInBag, trueCoefs)
colnames(testData$X) <- paste0("X", 1:ncol(testData$X))
(instanceResponseTest <- as.numeric(with(trainData, tapply(Z, ID, any))))
pred_EST <- with(testData, predict(milrFit_EST, X, ID, type = "bag"))
# predict(milrFit_EST, testData$X, testData$ID, 
#         type = "instance") # instance-level prediction
table(DATA = instanceResponseTest, PRED = pred_EST) 

## ----VS, message=FALSE--------------------------------------------------------
set.seed(99)
# Set the new coefficienct vector (large p)
trueCoefs_Lp <- c(-2, -2, -1, 1, 2, 0.5, rep(0, 45))
# Generate the new training data with large p
trainData_Lp <- DGP(numOfBag, numOfInstsInBag, trueCoefs_Lp)
colnames(trainData_Lp$X) <- paste0("X", 1:ncol(trainData_Lp$X))
# variable selection by user-defined tuning set
lambdaSet <- exp(seq(log(0.01), log(20), length = 20))
milrFit_VS <- with(trainData_Lp, milr(Z, X, ID, lambda = lambdaSet))
# grep the active factors and their corresponding coefficients
coef(milrFit_VS) %>>% `[`(abs(.) > 0)

## ----AUTOVS,message=FALSE-----------------------------------------------------
# variable selection using auto-tuning
milrFit_auto_VS <- milr(trainData_Lp$Z, trainData_Lp$X, trainData_Lp$ID,
                        lambda = -1, numLambda = 5) 
# the auto-selected lambda values
milrFit_auto_VS$lambda 
# the values of BIC under each lambda value
milrFit_auto_VS$BIC
# grep the active factors and their corresponding coefficients
coef(milrFit_auto_VS) %>>% `[`(abs(.) > 0)

## ----CV,message=FALSE---------------------------------------------------------
# variable selection using auto-tuning with cross validation
milrFit_auto_CV <- milr(trainData_Lp$Z, trainData_Lp$X, trainData_Lp$ID,
                        lambda = -1, numLambda = 5, 
                        lambdaCriterion = "deviance", nfold = 3) 
# the values of predictive deviance under each lambda value
milrFit_auto_CV$cv 
# grep the active factors and their corresponding coefficients
coef(milrFit_auto_CV) %>>% `[`(abs(.) > 0)

## ----DLMUSK1------------------------------------------------------------------
dataName <- "MIL-Data-2002-Musk-Corel-Trec9.tgz"
dataUrl <- "http://www.cs.columbia.edu/~andrews/mil/data/"

## ----READMUSK1----------------------------------------------------------------
filePath <- file.path(getwd(), dataName)
# Download MIL data sets from the url (not run)
# if (!file.exists(filePath))
#  download.file(paste0(dataUrl, dataName), filePath)
# Extract MUSK1 data file (not run)
# if (!dir.exists("MilData"))
#   untar(filePath, files = "musk1norm.svm")
# Read and Preprocess MUSK1
library(data.table)
MUSK1 <- fread("musk1norm.svm", header = FALSE) %>>%
  `[`(j = lapply(.SD, function(x) gsub("\\d+:(.*)", "\\1", x))) %>>%
  `[`(j = c("bag", "label") := tstrsplit(V1, ":")) %>>%
  `[`(j = V1 := NULL) %>>% `[`(j = lapply(.SD, as.numeric)) %>>%
  `[`(j = `:=`(bag = bag + 1, label = (label + 1)/2)) %>>%
  setnames(paste0("V", 2:(ncol(.)-1)), paste0("V", 1:(ncol(.)-2))) %>>%
  `[`(j = paste0("V", 1:(ncol(.)-2)) := lapply(.SD, scale), 
       .SDcols = paste0("V", 1:(ncol(.)-2)))
X <- paste0("V", 1:(ncol(MUSK1) - 2), collapse = "+") %>>% 
  (paste("~", .)) %>>% as.formula %>>% model.matrix(MUSK1) %>>% `[`( , -1L)
Y <- as.numeric(with(MUSK1, tapply(label, bag, function(x) sum(x) > 0)))

## ----MIFIT,message=FALSE,results="hide"---------------------------------------
# softmax with alpha = 0
softmaxFit_0 <- softmax(MUSK1$label, X, MUSK1$bag, alpha = 0, 
                        control = list(maxit = 5000))
# softmax with alpha = 3
softmaxFit_3 <- softmax(MUSK1$label, X, MUSK1$bag, alpha = 3, 
                        control = list(maxit = 5000))
# use a very small lambda so that milr do the estimation 
# without evaluating the Hessian matrix
milrFit <- milr(MUSK1$label, X, MUSK1$bag, lambda = 1e-7, maxit = 5000) 

## ----MILRVS, cache=TRUE,cache.lazy=FALSE,message=FALSE,warning=FALSE,tidy=FALSE----
# MILR-LASSO
milrSV <- milr(MUSK1$label, X, MUSK1$bag, lambda = -1, numLambda = 20, 
               nfold = 3, lambdaCriterion = "deviance", maxit = 5000)
# show the detected active covariates
sv_ind <- names(which(coef(milrSV)[-1L] != 0)) %>>% 
  (~ print(.)) %>>% match(colnames(X))
# use a very small lambda so that milr do the estimation 
# without evaluating the Hessian matrix
milrREFit <- milr(MUSK1$label, X[ , sv_ind], MUSK1$bag, 
                  lambda = 1e-7, maxit = 5000)
# Confusion matrix of the fitted model
table(DATA = Y, FIT_MILR = fitted(milrREFit, type = "bag"))

## ----MUSK1PRED2,message=FALSE-------------------------------------------------
set.seed(99)
predY <- matrix(0, length(Y), 4L) %>>%
  `colnames<-`(c("s0","s3","milr","milr_sv"))
folds <- 3
foldBag <- rep(1:folds, floor(length(Y) / folds) + 1, 
               length = length(Y)) %>>% sample(length(.))
foldIns <- rep(foldBag, table(MUSK1$bag))
for (i in 1:folds) {
  # prepare training and testing sets
  ind <- which(foldIns == i)
  # train models
  fit_s0 <- softmax(MUSK1[-ind, ]$label, X[-ind, ], MUSK1[-ind, ]$bag,
                    alpha = 0, control = list(maxit = 5000))
  fit_s3 <- softmax(MUSK1[-ind, ]$label, X[-ind, ], MUSK1[-ind, ]$bag,
                    alpha = 3, control = list(maxit = 5000))
  # milr, use a very small lambda so that milr do the estimation
  #       without evaluating the Hessian matrix
  fit_milr <- milr(MUSK1[-ind, ]$label, X[-ind, ], MUSK1[-ind, ]$bag,
                   lambda = 1e-7, maxit = 5000)
  fit_milr_sv <- milr(MUSK1[-ind, ]$label, X[-ind, sv_ind], MUSK1[-ind, ]$bag,
                      lambda = 1e-7, maxit = 5000)
  # store the predicted labels
  ind2 <- which(foldBag == i)
  # predict function returns bag response in default
  predY[ind2, 1L] <- predict(fit_s0, X[ind, ], MUSK1[ind, ]$bag)
  predY[ind2, 2L] <- predict(fit_s3, X[ind, ], MUSK1[ind, ]$bag)
  predY[ind2, 3L] <- predict(fit_milr, X[ind, ], MUSK1[ind, ]$bag)
  predY[ind2, 4L] <- predict(fit_milr_sv, X[ind, sv_ind], MUSK1[ind, ]$bag)
}

table(DATA = Y, PRED_s0 = predY[ , 1L])
table(DATA = Y, PRED_s3 = predY[ , 2L])
table(DATA = Y, PRED_MILR = predY[ , 3L])
table(DATA = Y, PRED_MILR_SV = predY[ , 4L])

