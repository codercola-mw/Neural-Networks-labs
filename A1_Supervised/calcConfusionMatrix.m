function [ cM ] = calcConfusionMatrix( LPredTest, LTest )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTest);
NClasses = length(classes);

% Add your own code here
cM = zeros(NClasses);
cM = confusionmat(LPredTest, LTest);
end

%done