clear
clc

n_train = 20;
n_test = 100;
d = 8;

%% Generate Data
[xtest,ytest,xtrain,ytrain,xsource,ysource] = gen_data(n_train,n_test,d);

%% Building MIST model
MIST_model = mist(xtrain,ytrain,xsource,ysource);
yhat_mist = MIST_model.predict(xtest);
rmse_MIST = sqrt(mse(yhat_mist-ytest));

gp_model = fitrgp(xtrain,ytrain);
yhat_gp = gp_model.predict(xtest);
rmse_gp = sqrt(mse(yhat_gp,ytest));

rmse_MIST
rmse_gp

