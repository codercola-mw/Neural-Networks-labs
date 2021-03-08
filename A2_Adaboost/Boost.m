function [idm, pm, tm, am] = Boost(xTrain,M,yTrain)
% inputs:
%   -M : number of classifier
%   -xTrain : input training data
%   -yTrain : output training data

% output:
%   pm: polarity in min error
%   tm: threshold 
%   am: alpha, performance of each classifier
%   idm: the id of each classifier

n = size(xTrain, 2);
d = ones(1, n)/n; % inital weight
%M = 6; %number of classifier
pm = zeros(1,M);
tm = zeros(1,M);
em = ones(1,M)*inf;   % min error
am = zeros(1,M);
idm = zeros(1,M);

%iter the classifier
for m = 1:M 
    best_c = 0;
    %iter the features
    for f = 1:size(xTrain,1) 
        %iter the thresholds
        for t= 1:M
            threshold = xTrain(f,t);
            p = 1;
            C = WeakClassifier(threshold,p,xTrain(f,:));         
            E = WeakClassifierError(C, d, yTrain);
            if E > 0.5
                E = 1-E;
                p = -1;
                C = -C; 
            end
            %minimizes the error
            if E < em(m)
                em(m) = E;
                pm(m) = p;
                tm(m) = t;
                idm(m) = f;
                best_c = C;
   
            end
        end
    end

    am(m) = 0.5*real(log((1-em(m))/em(m) ));
    d = d.* exp(-am(m)* (yTrain.*best_c));
    d = d/sum(d);
end

end
    