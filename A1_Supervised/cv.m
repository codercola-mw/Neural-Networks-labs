function [accuracy, k] = cv(X, L, D,nfold, k_range)

% split the data into nfolds
%len = size(X,1);
%ind = floor(len/nfold);

NBins = nfold;
selectAtRandom = true;
NSamplesPerLabelPerBin = inf;
[Xbin,Dbin, Lbin] = selectTrainingSamples(X,D,L, NSamplesPerLabelPerBin, NBins, selectAtRandom);

for k = 1:k_range

    i=1;
    for n = 1:nfold
        
        XX = Xbin;
        LL = Lbin;
        
        Xtest = combineBins(XX(n),1);
        Ltest = combineBins(LL(n),1);
        
        XX(n) = [];
        LL(n) = [];
        
        Xtrain = combineBins(XX,1:nfold-1);
        Ltrain = combineBins(LL,1:nfold-1);
     
     %%
     % Classify training data
        LPredtest = kNN(Xtest, k, Xtrain, Ltrain);
     
      % The confucionMatrix
        cM = calcConfusionMatrix(LPredtest, Ltest);
        acc = calcAccuracy(cM);
        
        accs(i) = acc;
        i=i+1;
                 
    end
    
    acc_K(k) = mean(accs);
       
    end

accuracy = acc_K;
k = [1:k_range];
end
