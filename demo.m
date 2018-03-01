clear
clc

n_train = 20;
n_test = 100;
n_source = 200;
d = 8;

%% Generate Data
xtest = lhsdesign(n_test,d);
xtrain = lhsdesign(n_train,d);
xsource = lhsdesign(n_source,d);

ytest = mean(dtlz1b(xtest,[30,3]),2);
ytrain= mean(dtlz1b(xtrain,[30,3]),2);
ysource = mean(dtlz1b(xsource,[0,0]),2);

%% Building MIST model
MIST_model = mist(xtrain,ytrain,xsource,ysource);
yhat_mist = MIST_model.predict(xtest);
rmse_MIST = sqrt(mse(yhat_mist-ytest));

gp_model = fitrgp(xtrain,ytrain);
yhat_gp = gp_model.predict(xtest);
rmse_gp = sqrt(mse(yhat_gp,ytest));

rmse_MIST
rmse_gp

