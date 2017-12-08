seismic.Rmd
================
Shravan Kuchkula, Dave Dyer, Tommy Pompo
12/03/2017

-   [Introduction](#introduction)
-   [Getting the data](#getting-the-data)
-   [Data set description](#data-set-description)
    -   [Variable Types and Cardinality](#variable-types-and-cardinality)
-   [Exploratory Data Analysis](#exploratory-data-analysis)
    -   [How are nbumps distributed ?](#how-are-nbumps-distributed)
    -   [Logistic Regression Assumptions](#logistic-regression-assumptions)
-   [Problem with unbalanced class variable](#problem-with-unbalanced-class-variable)
-   [Logistic Regression using cv.glmnet with balanced dataset with train/test split](#logistic-regression-using-cv.glmnet-with-balanced-dataset-with-traintest-split)
-   [Evaluating model performance using Cross Validation](#evaluating-model-performance-using-cross-validation)
-   [Comparing the performance of classification techniques.](#comparing-the-performance-of-classification-techniques.)
-   [References](#references)

Introduction
------------

The dangers associated with coal mining are myriad; black lung, flammable gas pockets, rockbursts, and tunnel collapses are all very real dangers that mining companies must consider when attempting to provide safe working conditions for miners. One class of mining hazard, commonly called 'seismic hazards', are notoriously difficult to protect against and even more difficult to predict with certainty. Therefore, predicting these hazards has become a well-known problem for machine learning and predictive analytics. The UCI Machine Learning Repository (<https://archive.ics.uci.edu>) provides a 'seismic bumps' data set that contains many records of combined categorical and numeric variables that could be used to predict seismic hazards. This 'seismic bumps' data set can be found at <https://archive.ics.uci.edu/ml/datasets/seismic-bumps>.

Our analysis attempts to use logistic regression techniques to predict whether a seismic 'bump' is predictive of a notable seismic hazard. We attempt to characterize our prediction accuracy and compare the results against the state of the art results from other statistical and machine learning techniques, that are included within the data set.

Getting the data
----------------

``` r
source("libraries.R")
#url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff"
#download.file(url, "seismic-bumps.arff")
seismicData <- import("seismic-bumps.arff")
```

Data set description
--------------------

The data were taken from from instruments in the Zabrze-Bielszowice coal mine, in Poland. There are 2,584 records, with only 170 class = 1 variables, so the data are significantly skewed towards non-hazardous training data. Field descriptions are below, but essentially energy readings and bump counts during one work shift are used to predict a 'hazardous' bump during the next shift. From the data description, a 'hazardous bump' is a seismic event with &gt; 10,000 Joules, and a 'shift' is a period of 8 hours. For the sake of reference, a practical example of 10,000 Joules would be the the approximate energy required to lift 10,000 tomatoes 1m above the ground. A class = 1 variable result signifies that a harzardous bump did, indeed, occur in the following shift to the measured data. Here is an example of the fields in the data set.

    ## Observations: 2,584
    ## Variables: 19
    ## $ seismic        <fctr> a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a...
    ## $ seismoacoustic <fctr> a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, b...
    ## $ shift          <fctr> N, N, N, N, N, W, W, N, N, W, N, N, W, N, N, W...
    ## $ genergy        <dbl> 15180, 14720, 8050, 28820, 12640, 63760, 207930...
    ## $ gpuls          <dbl> 48, 33, 30, 171, 57, 195, 614, 194, 303, 675, 1...
    ## $ gdenergy       <dbl> -72, -70, -81, -23, -63, -73, -6, -27, 54, 4, -...
    ## $ gdpuls         <dbl> -72, -79, -78, 40, -52, -65, 18, -3, 52, 25, -3...
    ## $ ghazard        <fctr> a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a...
    ## $ nbumps         <dbl> 0, 1, 0, 1, 0, 0, 2, 1, 0, 1, 1, 0, 1, 1, 1, 2,...
    ## $ nbumps2        <dbl> 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 1, 0,...
    ## $ nbumps3        <dbl> 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 2,...
    ## $ nbumps4        <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ nbumps5        <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ nbumps6        <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ nbumps7        <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ nbumps89       <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
    ## $ energy         <dbl> 0, 2000, 0, 3000, 0, 0, 1000, 4000, 0, 500, 500...
    ## $ maxenergy      <dbl> 0, 2000, 0, 3000, 0, 0, 700, 4000, 0, 500, 5000...
    ## $ class          <fctr> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...

![](Seismic_files/figure-markdown_github/unnamed-chunk-2-1.png)

From the UCI Machine Learning Repository, these are the field descriptions:

-   **seismic**: result of shift seismic hazard assessment in the mine working obtained by the seismic method (a - lack of hazard, b - low hazard, c - high hazard, d - danger state);
-   **seismoacoustic**: result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method;
-   **shift**: information about type of a shift (W - coal-getting, N -preparation shift);
-   **genergy**: seismic energy recorded within previous shift by the most active geophone (GMax) out of geophones monitoring the longwall;
-   **gpuls**: a number of pulses recorded within previous shift by GMax;
-   **gdenergy**: a deviation of energy recorded within previous shift by GMax from average energy recorded during eight previous shifts;
-   **gdpuls**: a deviation of a number of pulses recorded within previous shift by GMax from average number of pulses recorded during eight previous shifts;
-   **ghazard**: result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming form GMax only;
-   **nbumps**: the number of seismic bumps recorded within previous shift;
-   **nbumps2**: the number of seismic bumps (in energy range \[10<sup>2,10</sup>3)) registered within previous shift;
-   **nbumps3**: the number of seismic bumps (in energy range \[10<sup>3,10</sup>4)) registered within previous shift;
-   **nbumps4**: the number of seismic bumps (in energy range \[10<sup>4,10</sup>5)) registered within previous shift;
-   **nbumps5**: the number of seismic bumps (in energy range \[10<sup>5,10</sup>6)) registered within the last shift;
-   **nbumps6**: the number of seismic bumps (in energy range \[10<sup>6,10</sup>7)) registered within previous shift;
-   **nbumps7**: the number of seismic bumps (in energy range \[10<sup>7,10</sup>8)) registered within previous shift;
-   **nbumps89**: the number of seismic bumps (in energy range \[10<sup>8,10</sup>10)) registered within previous shift;
-   **energy**: total energy of seismic bumps registered within previous shift;
-   **maxenergy**: the maximum energy of the seismic bumps registered within previous shift;
-   **class**: the decision attribute - '1' means that high energy seismic bump occurred in the next shift ('hazardous state'), '0' means that no high energy seismic bumps occurred in the next shift ('non-hazardous state').

### Variable Types and Cardinality

There are 18 input variables and one binary output variable ("class"). The data are mostly numeric with 4 categorical input variables. However, some of the numeric values only contain a handful of discrete values which can be viewed as coded categorical variables. In particular, maxenergy and the 'nbumps(n)' variables can be treated as categorical. So, in short, we see the following breakdown in variable types:

The categorical variables are seismic, seismoacoustic, shift, ghazard, nbumps, nbumps2, nbumps3, nbumps4, nbumps5, nbumps6, nbumps7, nbumps89, class and the continuous variables are genergy, gdpuls, energy, maxenergy. The output variable is 'class'.

A table outlining the variables and some of their attributes is below:

           variable Cardinality Nulls Total   Uniqueness Distinctness
              class           2     0  2584 0.0007739938 0.0007739938
             energy         242     0  2584 0.0936532508 0.0936532508
           gdenergy         334     0  2584 0.1292569659 0.1292569659
             gdpuls         292     0  2584 0.1130030960 0.1130030960
            genergy        2212     0  2584 0.8560371517 0.8560371517
            ghazard           3     0  2584 0.0011609907 0.0011609907
              gpuls        1128     0  2584 0.4365325077 0.4365325077
          maxenergy          33     0  2584 0.0127708978 0.0127708978
             nbumps          10     0  2584 0.0038699690 0.0038699690
            nbumps2           7     0  2584 0.0027089783 0.0027089783
            nbumps3           7     0  2584 0.0027089783 0.0027089783
            nbumps4           4     0  2584 0.0015479876 0.0015479876
            nbumps5           2     0  2584 0.0007739938 0.0007739938
            nbumps6           1     0  2584 0.0003869969 0.0003869969
            nbumps7           1     0  2584 0.0003869969 0.0003869969
           nbumps89           1     0  2584 0.0003869969 0.0003869969
            seismic           2     0  2584 0.0007739938 0.0007739938
     seismoacoustic           3     0  2584 0.0011609907 0.0011609907
              shift           2     0  2584 0.0007739938 0.0007739938

Exploratory Data Analysis
-------------------------

It is important to understand how many observations are "hazardous state (class = 1)" and "non-hazardous state (class = 0)"

``` r
table(seismicData$class)
```

    ## 
    ##    0    1 
    ## 2414  170

As mentioned above, the data set output variable is highly skewed, and contains many more non-hazardous classes than it does hazardous classes.

### How are nbumps distributed ?

Distribution of all nbumps: Use `cowplot` to display all nbumps in a grid. ![](Seismic_files/figure-markdown_github/unnamed-chunk-7-1.png)

### Logistic Regression Assumptions

#### Linearity

#### Independence of Errors

#### Multi-collinearity

Collect all the numeric variables and check for multi-collinearity:

``` r
seismicDataNumeric <- seismicData %>%
  select(genergy, gpuls, gdenergy, gdpuls, energy, maxenergy)
```

``` r
# Create the correlation matrix
M <- round(cor(seismicDataNumeric), 2)

# Create corrplot
corrplot(M, method="pie", type = "lower")
```

![](Seismic_files/figure-markdown_github/unnamed-chunk-9-1.png)

Problem with unbalanced class variable
--------------------------------------

<Explain the problem we faced while using the dataset as is>

Logistic Regression using cv.glmnet with balanced dataset with train/test split
-------------------------------------------------------------------------------

We will first balance the class variable so that there are approximately equal number of hazardous vs non-hazardous cases.

``` r
# Reimport the dataset and start fresh
sb <- import("seismic-bumps.arff")

# Extract non-hazardous observations
nonh <- sb %>%
  filter(class == 0)

# Randomly choose 200 observations
nonh <- nonh[sample(1:nrow(nonh), 200), ]

# Extract hazardous observations
h <- sb %>%
  filter(class == 1)

# Combine the nonh and h dataframes
balancedSB <- rbind(nonh, h)

# Shuffle the dataframe
balancedSB <- balancedSB[sample(1:nrow(balancedSB)),]
```

Next, we will split the dataset into training (75%) and test (25%) datasets.

``` r
# Calculate N
N <- nrow(balancedSB)

# Create a random number vector
rvec <- runif(N)

# Select rows from the dataframe
balancedSB.train <- balancedSB[rvec < 0.75,]
balancedSB.test <- balancedSB[rvec >= 0.75,]

# Select rows for the class variable
train_class <- balancedSB.train$class
test_class <- balancedSB.test$class
```

Next, we build the model.

``` r
#Build the formula for model.matrix
formula <- as.formula("class ~ .")

#Build the model matrix object from the training data
balancedSBmatrix <- model.matrix(formula, data = balancedSB.train)

# Pass the matrix and class vector
cvfit <- cv.glmnet(balancedSBmatrix, train_class, family = "binomial", type.measure = "class")

# plot the model
plot(cvfit)
```

![](Seismic_files/figure-markdown_github/unnamed-chunk-12-1.png)

We are essentially looking for the lamba value which yeilds the lowest miss-classification error. That lamba value will be the number of coefficients that are contributing the most.

Check the coefficients

``` r
coef(cvfit, s = "lambda.min")
```

    ## 22 x 1 sparse Matrix of class "dgCMatrix"
    ##                             1
    ## (Intercept)     -0.6877404393
    ## (Intercept)      .           
    ## seismicb         0.0412656255
    ## seismoacousticb  .           
    ## seismoacousticc  .           
    ## shiftW           0.0011580912
    ## genergy          .           
    ## gpuls            0.0003433553
    ## gdenergy         .           
    ## gdpuls           .           
    ## ghazardb         .           
    ## ghazardc         .           
    ## nbumps           0.2048022200
    ## nbumps2          .           
    ## nbumps3          .           
    ## nbumps4          .           
    ## nbumps5          .           
    ## nbumps6          .           
    ## nbumps7          .           
    ## nbumps89         .           
    ## energy           .           
    ## maxenergy        .

**Prediction**:

We will use the above model to predict on the test dataset.

``` r
# Since predict function expects a matrix to be passed to newx
# Note the use of balancedSB.test here instead of train
balancedSBmatrix <- model.matrix(formula, data = balancedSB.test)

# predict on the test data, use type = "response" to get prediction probabilities
pred <- predict(cvfit, newx = balancedSBmatrix, type = "response")

# create a prediction object
predObj <- prediction(pred, test_class)
```

**Prediction Performance**:

``` r
# Measure the performance using true positive rate vs false positive rate
myroc.perf <- performance(predObj, measure = "tpr", x.measure = "fpr")

# Measure the performance using AUC
auc.test <- performance(predObj, measure = "auc")

# Get the AUC value to display on the ROC plot
auc.value <- auc.test@y.values

# Plot the ROC with AUC value
plot(myroc.perf)
abline(a=0, b= 1) #Ref line indicating poor performance
text(x = .40, y = .6,paste("AUC = ", round(auc.value[[1]],3), sep = ""))
```

![](Seismic_files/figure-markdown_github/unnamed-chunk-15-1.png)

Note: The effect of balancing the class variable can be observed in the below table:

200 -&gt; AUC was close to 80% 500 -&gt; AUC was close to 75% 1000 -&gt; AUC was close to 50%

This shows that logistic regression model's performance is largely dependent on having a balanced class variable.

Evaluating model performance using Cross Validation
---------------------------------------------------

Using cross validation we can assess how well our model building process works. The idea is that we can know how well our model will perform on new data not yet collected. We will use AUC as the performance metric.

``` r
# Reimport the dataset and start fresh
sb <- import("seismic-bumps.arff")

# Extract non-hazardous observations
nonh <- sb %>%
  filter(class == 0)

# Randomly choose 200 observations
nonh <- nonh[sample(1:nrow(nonh), 200), ]

# Extract hazardous observations
h <- sb %>%
  filter(class == 1)

# Combine the nonh and h dataframes
balancedSB <- rbind(nonh, h)

# Shuffle the dataframe
balancedSB <- balancedSB[sample(1:nrow(balancedSB)),]

# Extract the class variable into a vector call it "dat.train.y"
dat.train.y <- balancedSB$class

# Create a model matrix using the entire balancedSB. 
# Earlier we split this into training/test data. 
# Now we are going do that within the loop. 

dat.train.x <- model.matrix(as.formula("class ~ ."), data = balancedSB)
```

Now we are ready for the cross-validation loop

``` r
# Define the number of cross-validation loops
nloops <- 50

# Number of observations in the training (i.e full) dataset
ntrains <- nrow(dat.train.x)

# Create an empty vector to hold all the AUC result.
# Later we will display each run's AUC in a histogram
cv.aucs <- c()
```

Run the cross validation

``` r
for (i in 1:nloops){
  
 # Get the indexes using sample
 index<-sample(1:ntrains,60)
 
 # randomly draw 60 observations from front and call it training set
 cvtrain.x<-as.matrix(dat.train.x[index,])
 
 # randomly draw 60 observations from back and call it test set
 cvtest.x<-as.matrix(dat.train.x[-index,])
 
 # Get the corresponding class variable
 cvtrain.y<-dat.train.y[index]
 cvtest.y<-dat.train.y[-index]
 
 ## Model fitting:
 # Call cv.glmnet to fit the model using training set
 cvfit <- cv.glmnet(cvtrain.x, cvtrain.y, family = "binomial", type.measure = "class")
 
 ## Prediction:
 # Predict using test set using the above model. Use type = "response" to get prediction probabilities.
 fit.pred <- predict(cvfit, newx = cvtest.x, type = "response")
 # Prediction function takes prediction probabilities as first arg and class to compare as its second arg
 pred <- prediction(fit.pred[,1], cvtest.y)
 
 ## Prediction Performance:
 # Check prediction performance
 roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
 auc.train <- performance(pred, measure = "auc")
 auc.train <- auc.train@y.values
 
 # Store the auc value for this run into the vector
 cv.aucs[i]<-auc.train[[1]]

}
```

Draw a historgram of cv.aucs

``` r
hist(cv.aucs)
```

![](Seismic_files/figure-markdown_github/unnamed-chunk-19-1.png)

This indicates that a majority of time our model prediction performance lies between 70 to 75%

Comparing the performance of classification techniques.
-------------------------------------------------------

Within the ARFF file is a large table of various classification techniques and their results.

Classification results using stratified 10-fold cross-validation repeated 10 times

| Algorithm                   | Acc.      | BAcc.     | Acc.0 spec | Acc.1 sense | Size  |
|-----------------------------|-----------|-----------|------------|-------------|-------|
| q-ModLEM(entropy-RSS) (1)   | 80.2(5.1) | 69.1(6.2) | 81.90      | 56.35       | 27.5  |
| q-ModLEM(entropy-Corr.) (1) | 82.9(4.5) | 67.9(7.2) | 85.15      | 50.65       | 45.5  |
| MODLEM (2)                  | 92.5(0.8) | 52.6(2.8) | 98.58      | 6.65        | 145.5 |
| MLRules(-M 30) (3)          | 93.2(0.3) | 50.5(1.3) | 99.69      | 1.29        | 30    |
| MLRules(-M 100) (3)         | 92.9(0.6) | 52.0(2.2) | 99.10      | 4.88        | 100   |
| MLRules(-M 500) (3)         | 92.3(0.6) | 52.9(2.8) | 98.27      | 7.59        | 500   |
| BRACID (4)                  | 87.5(0.4) | 62.0(2.6) | 91.38      | 32.71       | -     |
| Jrip (Weka)                 | 93.0(0.6) | 51.4(2.4) | 99.35      | 3.47        | 1.8   |
| PART (Weka)                 | 92.1(0.8) | 52.7(3.5) | 98.09      | 7.35        | 34    |
| J48 (Weka)                  | 93.1(0.8) | 50.2(0.9) | 99.64      | 0.82        | 5.6   |
| SimpleCart (Weka)           | 93.4(0.0) | 50.0(0.0) | 100        | 0.00        | 1.0   |
| NaiveBayes (Weka)           | 86.7(2.0) | 64.7(5.8) | 90.08      | 39.41       | -     |
| IB1 (Weka)                  | 89.4(1.6) | 55.3(4.8) | 94.54      | 16.06       | -     |
| RandomForest(-I 100) (Weka) | 93.1(0.6) | 52.1(2.5) | 99.31      | 4.88        | 100   |

------------------------------------------------------------------------

References
----------

-   [Application of rule induction algorithms for analysis of data collected by seismic hazard monitoring systems in coal mines.](https://actamont.tuke.sk/pdf/2013/n4/7sikora.pdf)
-   [A Study of Rockburst Hazard Evaluation Method in Coal Mine](https://www.hindawi.com/journals/sv/2016/8740868/#B13)
-   [Classification: Basic concepts, decision trees and model evaluation](https://www-users.cs.umn.edu/~kumar001/dmbook/ch4.pdf)
