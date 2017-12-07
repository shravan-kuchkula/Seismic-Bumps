# Load all the libraries
installRequiredPackages <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[,"Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE, repos='http://cran.us.r-project.org')
  sapply(pkg, require, character.only = TRUE)
}

libs <- c("readr", "dplyr", "tidyr", "ggplot2",
          "magrittr", "markdown", "knitr", "yaml",
          "corrplot", "GGally", "broom", "psych",
          "car", "vtreat", "caret", "mlbench",
          "caTools", "rio", "ranger", "pROC",
          "reshape", "cowplot", "glmnet", "ROCR"
          )

installRequiredPackages(libs)

