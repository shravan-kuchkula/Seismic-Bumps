# Load all the libraries
installRequiredPackages <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[,"Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

libs <- c("readr", "dplyr", "tidyr", "ggplot2",
          "magrittr", "markdown", "knitr", "yaml",
          "corrplot", "GGally", "broom", "psych",
          "car", "vtreat", "caret", "mlbench",
          "caTools", "rio", "ranger", "pROC"
          )

installRequiredPackages(libs)

