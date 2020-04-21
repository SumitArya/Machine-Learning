set.seed(123)

## load the required libraries
library(dplyr)
library(caret)
library(psych)

## load the pro state cancer data
PSA_data <- read.csv("PSA", sep="")

## keep the needed predictors
PSA_data <- PSA_data %>% select(PSA,cancer_volume,prostate_weight,patient_age)

## ----------------------  Step 1: Exploratory Data Analysis  -------------
## Correlations between response and the potential predictors
pros.cor = cor(PSA_data)
round(pros.cor,3)
## we can see from above result PSA is high correlated with volume and 
## than weight and age is less correlated with negative effect.

## we can also visualise the same data
pairs.panels(PSA_data)
## PSA is more corelated to volume and weight, as visible from the graphs. 

## summary of the data
summary(PSA_data)

## ------------------- Step 2: Variable transformation -----------------------------
## PSA, volume and weight are right skewed, take logirthm to make them slightly normalise
PSA_data[,c("lpsa","lcv","lpw")]<-apply(PSA_data[,c("PSA","cancer_volume","prostate_weight")],2,log)

## select the required column and rename the patient_age column
PSA_data <- PSA_data %>% select(lpsa,lcv,lpw,patient_age) %>% rename(age=patient_age)

## divide the data as traning and testing
inTrain<-createDataPartition(PSA_data$lpsa,p=.75,times = 1,list = FALSE)
training=PSA_data[inTrain,]
testing=PSA_data[-inTrain,]

##--------------------  Step 3: Model selection   ------------------------------------
# Volume and weight seems related and could be one choice of model
modFit1<-lm(lpsa~ lcv ,data=training)
modFit2<-lm(lpsa~ lcv+lpw,data=training)
modFit3<-lm(lpsa~ lcv+lpw+age,data=training)
modFit4<-lm(lpsa~ lcv+lpw*age,data=training)
modFit5<-lm(lpsa~ lcv*lpw+age,data=training)

## get the fits
summary(modFit1)
summary(modFit2)
summary(modFit3)
summary(modFit4)
summary(modFit5)

## 2nd model is defining the 53.88% variability.

## --------------- Step 4: examine the residuals and check for influential cases-----------
## diagnostic tests for our regression
pros.fits = fitted(modFit2) # Vector of fitted values
plot(pros.fits, training$lpsa, main="Actual versus fitted values",
     xlab="Fitted values", ylab="Log PSA values") 
abline(a=0, b=1, lty=2, col="red") 

## some handy shortcuts, make intercept 0 and check the performance
modFit6<-lm(lpsa ~ 0 + lcv + lpw,data=training)
summary(modFit6)
## it is now explaining ~92.52% of variability which is better than the origional

## lets predict the outliers
# identify the leverage point outliers
lev = hatvalues(modFit6)
# plot
plot(lev, pch="*", cex=2, main="Influential Obs by leverage Point")  # plot cook's distance
abline(h = (2*(2+1)/nrow(training)), col="red")  # add cutoff line
text(x=1:length(lev)+1, y=lev, labels=ifelse(lev>(2*(2+1)/nrow(training)),names(lev),""), col="red")

## get the indexes of outliers
id.lev <- which(lev > (2*(2+1)/nrow(training)))
# 1 outlier

## cooks distance
cooksd = cooks.distance(modFit6)

## plot
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")

id.cooks <- which(cooksd > (4/nrow(training)))
# 4 outliers

## all outliers
idx<-c(id.cooks,id.lev)
names(idx)<-NULL

## remove these from traing and append to predict these values
testing<-rbind(testing,training[idx,])
training<-training[-idx,]

# Step 5:  ----------------- Re-run the model -------------
modFit6<-lm(lpsa ~ 0 + lcv + lpw,data=training)
summary(modFit6)

## it is now explaining 93.82% of variability which is better than the previous model

## fitted vs residual
qplot(fitted(modFit6),residuals(modFit6),data=training)
plot(residuals(modFit6),pch=19)  ## varies close to zero on both sides

## actual Vs fitted value
#qplot(lpsa,predictions6,data=testing)

# lets predict the values for the test data using this model
predictions6<-predict(modFit6,testing)

## examine the prediction performance factors
print_metrics = function(lin_mod, df, score){
  resids = df$lpsa - score
  resids2 = resids**2
  N = length(score)
  r2 = as.character(round(summary(lin_mod)$r.squared, 4))
  adj_r2 = as.character(round(summary(lin_mod)$adj.r.squared, 4))
  cat(paste('Mean Square Error      = ', as.character(round(sum(resids2)/N, 4)), '\\n'))
  cat(paste('Root Mean Square Error = ', as.character(round(sqrt(sum(resids2)/N), 4)), '\\n'))
  cat(paste('Mean Absolute Error    = ', as.character(round(sum(abs(resids))/N, 4)), '\\n'))
  cat(paste('Median Absolute Error  = ', as.character(round(median(abs(resids)), 4)), '\\n'))
  cat(paste('R^2                    = ', r2, '\\n'))
  cat(paste('Adjusted R^2           = ', adj_r2, '\\n'))
}

print_metrics(modFit6, testing, predictions6) 

## Step 4:
## kernel density plot and histogram of the residuals of the regression model
hist_resids = function(df, score, bins = 10){
  options(repr.plot.width=4, repr.plot.height=3) # Set the initial plot area dimensions
  df$resids = df$lpsa - score
  bw = (max(df$resids) - min(df$resids))/(bins + 1)
  ggplot(df, aes(resids)) + 
    geom_histogram(binwidth = bw, aes(y=..density..), alpha = 0.5) +
    geom_density(aes(y=..density..), color = 'blue') +
    xlab('Residual value') + ggtitle('Histogram of residuals')
}

hist_resids(testing, predictions6)
# This histogram and the kernel density plot look approximately Normal, but with some deviations. 
# Overall, these residuals look reasonable for a real-world model.

## Quantile-Quantile Normal plot
resids_qq = function(df, score, bins = 10){
  options(repr.plot.width=4, repr.plot.height=3.5) # Set the initial plot area dimensions
  df$resids = df$lpsa - score
  ggplot() + 
    geom_qq(data = df, aes(sample = resids)) + 
    stat_qq_line(data = df, aes(sample = resids)) +
    ylab('Quantiles of residuals') + xlab('Quantiles of standard Normal') +
    ggtitle('QQ plot of residual values')
}
# If the residuals were perfectly Normally distributed, these points would fall on a straight line. 
# In real-world problems, you should expect the straight line relationship to be approximate.
# seen from the diagnostic plots, all assumptions for inferential purpose has been satisfied.
resids_qq(testing, predictions6)

# Step 6: -----------
## Note: to test with new data, convert vol, weight in log. 
# value will also be in log..take antilog to get actual value
# Increased levels of PSA are linked with prostate cancer. We have not used Age as predictor in our final model.
# some values like psa, weight and volume are transformed to logarithmic in our model. to predict from our model
# you have to take these respective new values on log scale. 
