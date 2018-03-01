classdef mist
% MIST is an instance selection algorithm, that aims to select the
% optimal subset of source instances that, when concatenated with target 
% data, leads to the best predictive performance. The algorithm uses UMDA 
% for instance selection, a linear rescaling of source data to maximize 
% source-target similarity, and FITRGP as the base model.
% 
% MIST takes as inputs (x_target,y_target,x_source,y_source)
% Accepted inputs are shaped (n,d); n = # of instances, d = dimensions
% Accepted outputs are shaped (n,1)
% 
% Example Use:
% model = mist(x_target,y_target,x_source,y_source)
% yhat = model.predict(x_test)

    properties
        model
        best_selection
    end
    
    methods
        function o = mist(x,y,xs,ys)
        % Training the MIST model, 
        % Steps consists of Linear Rescaling of Source Data, and Instance Selection
        % Base model is FitRGP
            data.x = x;
            data.y = y;
            data.xs = xs;
            data.ys = ys;
            
            [o.best_selection,data] = umda(data);             
            x = [data.x;data.xs(o.best_selection,:)];
            y = [data.y;data.rescaled_ys(o.best_selection,:)];
            o.model = fitrgp(x,y);
        end
        
        function [yhat,sigma] = predict(o,x)
            %% Prediction
            [yhat,sigma] = o.model.predict(x);
        end
    end
end

function [best_selection,data] = umda(data)
    %% Running the UMDA algorithm for MIST
    
    gen.popsize = 20;
    maxiter = 50;
    
    data = rescale(data);
    nsrc = size(data.ys,1);
    %% Initialize
    gen.pop = logical(randi([0,1],gen.popsize,nsrc));
    gen.pop(1,:) = true(1,nsrc);
    gen.pop(2,:) = false(1,nsrc);
    for i = 1:gen.popsize
        gen.fitness_pop(i) = evaluate(data,gen.pop(i,:));
    end
    
    for iter = 1:maxiter
        
        %% Calculate Univariate Probability Vector, v
        gen.v = calc_v(gen.pop);
        
        %% Gen childpop
        rndvec = rand(gen.popsize,nsrc);
        gen.childpop = logical(rndvec);
        for i = 1:gen.popsize
            gen.childpop(i,:) = logical(rndvec(i,:)<gen.v);
            gen.fitness_childpop(i) = evaluate(data,gen.childpop(i,:));
        end
        
        %% Elitist Selection
        gen = elitist_selection(gen);
        
        fprintf('Iter %i - best so far: %.2d\n',iter,gen.fitness_pop(1));
    end    
    best_selection = gen.pop(1,:);
end

function data = rescale(data)       
    %% Linearly Rescale Source Outputs to match Target Outputs
    ntar = length(data.y);
    nsrc = length(data.ys);
    
    model = fitrgp(data.xs,data.ys);
    ysrc_at_xtar = model.predict(data.x);
    A = [ysrc_at_xtar ones(ntar,1)];
    b = data.y;
    
    data.scaling_coeff = A\b;
    data.rescaled_ys = [data.ys ones(nsrc,1)]*data.scaling_coeff;    
    
end

function v = calc_v(pop)
    %% Calculate Univariate Probability Vector, v
    v = sum(pop,1)/size(pop,1);
end

function gen = elitist_selection(gen)
    %% Select popsize best individuals
    combined_pop = [gen.pop;gen.childpop];
    fitness = [gen.fitness_pop,gen.fitness_childpop];
    [sorted_fitness,ind] = sort(fitness,'ascend');
    gen.pop = combined_pop(ind(1:gen.popsize),:);
    gen.fitness_pop = sorted_fitness(1:gen.popsize);
end

function fitness=evaluate(data,ind)
    %% Evaluate Individuals using 3-fold CV with FitRGP
    n = size(data.x,1);
    k=3;
    cv = cvpartition(n,'Kfold',k);
    
    rmse = zeros(1,3);
    for i = 1:k
        x = [data.x(cv.training(i),:);data.xs(ind,:)];
        y = [data.y(cv.training(i),:);data.rescaled_ys(ind,:)];
        
        model = fitrgp(x,y);
        yhat = model.predict(data.x(cv.test(i),:));
        rmse(i) = sqrt(mse(yhat-data.y(cv.test(i))));
    end
    fitness = mean(rmse);
end