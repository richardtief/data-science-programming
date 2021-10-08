library(Rcpp)

code <- "
# include <math.h>
double ed_c(NumericVector x, NumericVector y) {
 // your code
}"
cppFunction(code) # ed_c can be used in R

