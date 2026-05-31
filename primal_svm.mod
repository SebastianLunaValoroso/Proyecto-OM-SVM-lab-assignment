param n >= 0;
param m >= 0;

set N := 1..n;
set M := 1..m;

param nu;		#hiperparametro a validar
param y {M};
param x {M,N};

var w {N};
var gamma;
var s {M} >= 0;

#Formulacion Primal

minimize primal_svm: 0.5 * sum {j in N} (w[j] * w[j]) - nu * sum {i in M} (s[i]);

subj to constr {i in M}: y[i] * (sum {j in N} (w[j] * x[i,j]) + gamma) + s[i] >= 1;