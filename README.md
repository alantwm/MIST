# MIST - Metaheuristic Instance Selection for Transfer

MIST is an instance selection algorithm, that aims to select the optimal subset of source instances that, when concatenated with target 
data, leads to the best predictive performance. The algorithm uses UMDA for instance selection, a linear rescaling of source data to maximize source-target similarity, and FITRGP as the base model.

MIST takes as inputs (x_target, y_target, x_source, y_source)

Accepted inputs are shaped (n, d), and outputs are shaped (n, 1), where n = # of instances, d = dimensions

Example:
```matlab
model = mist(x_target,y_target,x_source,y_source)
yhat = model.predict(x_test)
```

Note: Scripts were tested in Matlab R2015b.

Described in detail in:
Min, Alan Tan Wei, et al. "Knowledge transfer through machine learning in aircraft design." IEEE Computational Intelligence Magazine 12.4 (2017): 48-60.
