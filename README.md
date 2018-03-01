# MIST

MIST is an instance selection algorithm, that aims to select the optimal subset of source instances that, when concatenated with target 
data, leads to the best predictive performance. The algorithm uses UMDA for instance selection, a linear rescaling of source data to maximize source-target similarity, and FITRGP as the base model.

MIST takes as inputs (x_target, y_target, x_source, y_source)

Accepted inputs are shaped (n, d), and outputs are shaped (n, 1), where n = # of instances, d = dimensions

Example:
```matlab
model = mist(x_target,y_target,x_source,y_source)
yhat = model.predict(x_test)
```

