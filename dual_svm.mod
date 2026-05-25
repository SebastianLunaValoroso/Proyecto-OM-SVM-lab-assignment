param m >= 0;

set M := 1..m;

param nu;
param y {M};
param K {M,M};

var lambda {M} >= 0, <= nu;

#Formulacion Dual

maximize dual_svm: sum {i in M} (lambda[i]) - 0.5 * sum {i in M}(sum {j in M} (lambda[i] * y[i] * lambda[j] * y[j] * K[i,j]));

subj to constr: sum {i in M} (lambda[i] * y[i]) = 0;