Seismic-data-dictionary
================
Dave, Shravan, Tommy
November 27, 2017

-   [Seismic Data](#seismic-data)
    -   [Variable Types and Cardinality](#variable-types-and-cardinality)

Seismic Data
============

A description of the Seismic data set can be found here: <https://archive.ics.uci.edu/ml/datasets/seismic-bumps>

### Variable Types and Cardinality

One of the things we need to do to prepare for logistic regression using most ML libraries is to encode the categorical variables using dummy encoding or one-hot encoding. In order to do that, and not deal with a prohibitively wide data set, we need to understand the cardinality of the variables and classify them accordingly.

``` r
seismicSummary
```

    ##          variable Cardinality Filled Nulls Total Complete   Uniqueness
    ## 1           class           2   2584     0  2584        1 0.0007739938
    ## 2          energy         242   2584     0  2584        1 0.0936532508
    ## 3        gdenergy         334   2584     0  2584        1 0.1292569659
    ## 4          gdpuls         292   2584     0  2584        1 0.1130030960
    ## 5         genergy        2212   2584     0  2584        1 0.8560371517
    ## 6         ghazard           3   2584     0  2584        1 0.0011609907
    ## 7           gpuls        1128   2584     0  2584        1 0.4365325077
    ## 8       maxenergy          33   2584     0  2584        1 0.0127708978
    ## 9          nbumps          10   2584     0  2584        1 0.0038699690
    ## 10        nbumps2           7   2584     0  2584        1 0.0027089783
    ## 11        nbumps3           7   2584     0  2584        1 0.0027089783
    ## 12        nbumps4           4   2584     0  2584        1 0.0015479876
    ## 13        nbumps5           2   2584     0  2584        1 0.0007739938
    ## 14        nbumps6           1   2584     0  2584        1 0.0003869969
    ## 15        nbumps7           1   2584     0  2584        1 0.0003869969
    ## 16       nbumps89           1   2584     0  2584        1 0.0003869969
    ## 17        seismic           2   2584     0  2584        1 0.0007739938
    ## 18 seismoacoustic           3   2584     0  2584        1 0.0011609907
    ## 19          shift           2   2584     0  2584        1 0.0007739938
    ##    Distinctness
    ## 1  0.0007739938
    ## 2  0.0936532508
    ## 3  0.1292569659
    ## 4  0.1130030960
    ## 5  0.8560371517
    ## 6  0.0011609907
    ## 7  0.4365325077
    ## 8  0.0127708978
    ## 9  0.0038699690
    ## 10 0.0027089783
    ## 11 0.0027089783
    ## 12 0.0015479876
    ## 13 0.0007739938
    ## 14 0.0003869969
    ## 15 0.0003869969
    ## 16 0.0003869969
    ## 17 0.0007739938
    ## 18 0.0011609907
    ## 19 0.0007739938

Based solely on the cardinality of values, it would appear that at least 5 variables (energy, gdenergy, gdpuls, genergy, and gpuls) are too continuous to dummy encode. The rest of the varialbes are feasible (at least) for one-hot / dummy encoding. The actual categorical variables are listed with their levels below, but \#TODO: We will investigate treating numeric variables as categorical in about half of the remaining variables, so they can be used as predictor features for the 'class' outcome variable.

    ## List of 19
    ##  $ seismic       : chr [1:2] "a" "b"
    ##  $ seismoacoustic: chr [1:3] "a" "b" "c"
    ##  $ shift         : chr [1:2] "N" "W"
    ##  $ genergy       : NULL
    ##  $ gpuls         : NULL
    ##  $ gdenergy      : NULL
    ##  $ gdpuls        : NULL
    ##  $ ghazard       : chr [1:3] "a" "b" "c"
    ##  $ nbumps        : NULL
    ##  $ nbumps2       : NULL
    ##  $ nbumps3       : NULL
    ##  $ nbumps4       : NULL
    ##  $ nbumps5       : NULL
    ##  $ nbumps6       : NULL
    ##  $ nbumps7       : NULL
    ##  $ nbumps89      : NULL
    ##  $ energy        : NULL
    ##  $ maxenergy     : NULL
    ##  $ class         : chr [1:2] "0" "1"
