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
-   [Logistic Regression Model](#logistic-regression-model)
-   [Evaluating the model performance.](#evaluating-the-model-performance.)
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

Logistic Regression Model
-------------------------

Fit a logistic regression model with what you think could be contributing to the seismic hazard.

``` r
seismic_model <- glm(class ~ seismic + seismoacoustic + shift + ghazard,
                     data = seismicData, family = "binomial")

summary(seismic_model)
```

    ## 
    ## Call:
    ## glm(formula = class ~ seismic + seismoacoustic + shift + ghazard, 
    ##     family = "binomial", data = seismicData)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -0.6757  -0.4321  -0.3998  -0.1856   2.9426  
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)      -4.05310    0.25427 -15.940  < 2e-16 ***
    ## seismicb          0.42551    0.16421   2.591  0.00956 ** 
    ## seismoacousticb   0.05025    0.17474   0.288  0.77369    
    ## seismoacousticc   0.70002    0.67220   1.041  0.29770    
    ## shiftW            1.56663    0.26392   5.936 2.92e-09 ***
    ## ghazardb         -0.31343    0.32304  -0.970  0.33192    
    ## ghazardc        -14.38103  424.95947  -0.034  0.97300    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1253.8  on 2583  degrees of freedom
    ## Residual deviance: 1178.9  on 2577  degrees of freedom
    ## AIC: 1192.9
    ## 
    ## Number of Fisher Scoring iterations: 15

Making a binary prediction: We used the glm() function to build a logistic regression model of the `class` variable. As with many of R's machine learning methods, you can apply the `predict()` function to the model object to forecast future behavior. By default, predict() outputs predictions in terms of log odds unless `type = "response"` is specified. This converts the log odds to probabilities.

Because a logistic regression model estimates the probability of the outcome, it is up to you to determine the threshold at which the probability implies action. One must balance the extremes of being too cautious versus being too aggressive. For example, if we classify an observation which has a probability of being in class 1 as 99% or greater, then we may miss out on some observations that may indeed be class 1 but were classified as class 0. This balance is particularly important to consider for severely imbalanced outcomes, such as in this dataset where class 1 are relatively rare.

``` r
# make a copy
seismicDataPredictions <- seismicData

# Estimate the probability of class 1
seismicDataPredictions$prob <- predict(seismic_model, type = "response")
```

Find the actual probability of an observation to be in class 1.

``` r
mean(as.numeric(as.character(seismicData$class)))
```

    ## [1] 0.06578947

We will use this as our cut-off threshold.

``` r
seismicDataPredictions$pred <- ifelse(seismicDataPredictions$prob > 0.0657, 1, 0)
```

Now calculate the model accuracy:

``` r
mean(seismicDataPredictions$pred == seismicDataPredictions$class)
```

    ## [1] 0.4384675

This shows that the logistic regression model with all the factor variables made a correct prediction 44% of the time.

What would be the accuracy of the model if a model had simply predicted class 0 for each observation ?

``` r
seismicDataPredictions$predNull <- 0
mean(seismicDataPredictions$predNull == seismicDataPredictions$class)
```

    ## [1] 0.9342105

With an accuracy of 44% the model is actually performing worse than if it were to predict class 0 for every record.

This illustrates that "rare events" create challenges for classification models. When 1 outcome is very rare predicting the opposite can result in very high accuracy.

Calculate ROC Curves and AUC: The previous exercises have demonstrated that accuracy is a very misleading measure of model performance on imbalanced datasets. Graphing the model's performance better illustrates the tradeoff between a model that is overly agressive and one that is overly passive. Here we will create a ROC curve and compute the area under the curve (AUC) to evaluate the logistic regression model that we created above.

``` r
ROC <- roc(seismicDataPredictions$class, seismicDataPredictions$prob)
plot(ROC, col = "blue")
text(x = .42, y = .6,paste("AUC = ", round(auc(ROC), 2), sep = ""))
```

![](Seismic_files/figure-markdown_github/unnamed-chunk-16-1.png)

Dummy variables, missing data and interactions:

``` r
seismic_model <- glm(class ~ . , data = seismicData, family = "binomial")
summary(seismic_model)
```

    ## 
    ## Call:
    ## glm(formula = class ~ ., family = "binomial", data = seismicData)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.1035  -0.3637  -0.2827  -0.1781   3.0300  
    ## 
    ## Coefficients: (3 not defined because of singularities)
    ##                   Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)     -4.426e+00  2.695e-01 -16.425  < 2e-16 ***
    ## seismicb         3.079e-01  1.844e-01   1.670  0.09500 .  
    ## seismoacousticb  3.625e-02  1.866e-01   0.194  0.84596    
    ## seismoacousticc  5.049e-01  6.767e-01   0.746  0.45563    
    ## shiftW           8.186e-01  2.926e-01   2.798  0.00515 ** 
    ## genergy         -7.967e-07  4.561e-07  -1.747  0.08066 .  
    ## gpuls            9.443e-04  2.148e-04   4.397  1.1e-05 ***
    ## gdenergy        -1.106e-03  2.115e-03  -0.523  0.60094    
    ## gdpuls          -2.337e-03  2.855e-03  -0.819  0.41302    
    ## ghazardb        -9.745e-03  3.448e-01  -0.028  0.97746    
    ## ghazardc        -1.390e+01  4.250e+02  -0.033  0.97391    
    ## nbumps           3.862e+00  1.460e+00   2.645  0.00817 ** 
    ## nbumps2         -3.478e+00  1.465e+00  -2.374  0.01760 *  
    ## nbumps3         -3.452e+00  1.461e+00  -2.362  0.01819 *  
    ## nbumps4         -3.778e+00  1.522e+00  -2.482  0.01307 *  
    ## nbumps5         -2.076e+00  2.625e+00  -0.791  0.42915    
    ## nbumps6                 NA         NA      NA       NA    
    ## nbumps7                 NA         NA      NA       NA    
    ## nbumps89                NA         NA      NA       NA    
    ## energy          -9.632e-06  3.159e-05  -0.305  0.76042    
    ## maxenergy        1.771e-06  3.121e-05   0.057  0.95475    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1253.8  on 2583  degrees of freedom
    ## Residual deviance: 1073.2  on 2566  degrees of freedom
    ## AIC: 1109.2
    ## 
    ## Number of Fisher Scoring iterations: 15

Evaluating the model performance.
---------------------------------

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
