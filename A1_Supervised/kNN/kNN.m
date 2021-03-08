function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred  = zeros(size(X,1),1);


% Euclidean
for i = 1:size(X,1)
  
   for j = 1:size(XTrain,1)
     
       dist(j) = sqrt( sum( (X(i,:) - XTrain(j,:)).^2 ) );
        
      
   end
  
    [dist_sorted, index] = sort(dist);
    L_sorted = LTrain(index);
   
    nn = L_sorted(1:k);
    knn = mode(nn);
   
    LPred(i) = knn;
   
end
   LPred;
end

