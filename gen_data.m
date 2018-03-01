function [xtest,ytest,xtrain,ytrain,xsrc,ysrc] = gen_data(n_train,n_test,d)
    rng('default')
    
    ld = [0.5 0.5]; % Weight vector for Tchebycheff Aggregation
    b = [30 3];
    %% Training Data
    % Target Task
    xtrain = lhsdesign(n_train,d);
    ftrain = dtlz1b(xtrain,b);
    ytrain = aggregate(ftrain,ld); % Tchebycheff Aggregation
    
    %% Test Data
    % Task 1
    xtest = lhsdesign(n_test,d);
    ftest = dtlz1b(xtest,b);
    ytest = aggregate(ftest,ld); % Tchebycheff Aggregation
    
    %% Build Src_models
    % Src 2
    b = [0,0];
    n_srcdata = 300;
    xsrc = lhsdesign(n_srcdata,d);
    ftrain = dtlz1b(xsrc,b);
    ysrc = aggregate(ftrain,ld); % Tchebycheff Aggregation
    [ysrc,~] = normalize(ysrc,[]);    
    
    %% Normalize
    [ytest,ytrain] = normalize(ytest,ytrain);
end

function [ytest,ytrain] = normalize(ytest,ytrain)
    y=[ytest;ytrain];
    miny = min(y);
    maxy = max(y);
    y = (y-miny)/(maxy-miny);
    
    ntest = length(ytest);
    ytest = y(1:ntest);
    ytrain = y(ntest+1:end);
    
end

function y=aggregate(f,ld)
    % Tchebycheff Aggregation
    m = size(f,2);
    for i = 1:m
        tmp(:,i) = f(:,i)*ld(i);
    end
    y = max(tmp,[],2);
end