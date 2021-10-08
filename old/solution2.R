library(Rcpp)

code <- "
# include <math.h>
double ed(NumericVector x, NumericVector y) {
}"

cppFunction(code)
