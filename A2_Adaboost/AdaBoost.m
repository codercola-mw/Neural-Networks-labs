%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 100;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 1000;   
% Number of weak classifiers
nbrWeakClassifiers = 50;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));


%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
M = nbrWeakClassifiers;
d = ones(1, nbrTrainImages);
d = d/nbrTrainImages;  % inital weight

% find the best set of P, a, t, id with min error

e = zeros(1,M);
p = ones(1,M);
ts = zeros(1,M);
id = zeros(1,M);
a = zeros(1,M);

%   iter each classifier
for m = 1:M 
    min_a = 0;   % alpha
    min_t = 0;  % threshold
    min_e = inf; % error
    min_id = 0;
    best_c = 0;
    min_p = 1;
 
    %iter the features and each data point
    for f = 1:size(xTrain,1) 
        for t = 1:size(xTrain,2) 
            threshold = xTrain(f,t);
            C = WeakClassifier(threshold,p(m),xTrain(f,:));         
            E = WeakClassifierError(C, d, yTrain);
            if E > 0.5
                E = 1-E;
                p(m) = -p(m); % flipped
                C = -C;        % flipped
    
            end
            %minimizes the error and update
            if E < min_e
                min_e = E;
                min_p = p(m);
                min_t = threshold;
                min_id = f;
                best_c = C;
                
                min_a = 0.5*log((1-min_e)/min_e );
            end
        end
    end
    
    
    d = d.* exp(-min_a*(yTrain.*best_c));
    d = d./sum(d);
  
    
    e(m) = min_e;
    p(m) = min_p;
    ts(m) = min_t;
    id(m) = min_id;
    a(m) = min_a ; 
    %am(m) = 0.5*real(log((1-em(m))/em(m) ));
    
end


%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

% train data

classifier = zeros(nbrWeakClassifiers,nbrTrainImages);
for t = 1:nbrWeakClassifiers
    
    classifier(t,:) = a(t).*WeakClassifier(ts(t), p(t), xTrain(id(t),:));
    
end  

strongclassifier = sign(sum(classifier,1));
acc_train = sum(strongclassifier == yTrain)/size(yTrain,2);
acc_train % 0.9840

% test data
classifier = zeros(nbrWeakClassifiers,nbrTestImages);
for t = 1:nbrWeakClassifiers
        classifier(t,:) = a(t).*WeakClassifier(ts(t),p(t),xTest(id(t),:));
end
strongclassifier = sign(sum(classifier,1));
acc_test = sum(strongclassifier == yTest)/size(yTest,2);

acc_test %0.9426


%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

% train data

classifier = zeros(nbrWeakClassifiers,nbrTrainImages);
acc_train = [];
for num = 1:nbrWeakClassifiers
    for t = 1:num
         classifier(t,:) = a(t).*WeakClassifier(ts(t), p(t), xTrain(id(t),:));
         
    end
    strongclassifier = sign(sum(classifier,1));
    acc_train(num) = sum(strongclassifier == yTrain)/size(yTrain,2);
end   

plot(acc_train);


% test data
classifier = zeros(nbrWeakClassifiers,nbrTestImages);
acc_test = [];
for num = 1:nbrWeakClassifiers
    for t = 1:num
         classifier(t,:) = a(t).*WeakClassifier(ts(t), p(t), xTest(id(t),:));
         
    end
    strongclassifier = sign(sum(classifier,1));
    acc_test(num) = sum(strongclassifier == yTest)/size(yTest,2);
end   
hold;
plot(acc_test);

legend({'training','test'},'Location','northwest');


%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

[misclass_id] = find((yTest ~= strongclassifier)==0);
[classed_id] = find((yTest == strongclassifier)==0);
%% missclassified face

colormap gray
for k = 1:25
    subplot(5,5,k), imagesc(testImages(:,:,misclass_id(k))), axis image, axis off
end

%% missclasified non face
len = size(misclass_id,2);

colormap gray
for k=1:25
    subplot(5,5,k), imagesc(testImages(:,:,misclass_id(len-k))), axis image, axis off
end

%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.
%features = [ 7 , 21, 30, 36]; %most freq feat
colormap gray
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,id(k+25)),[-1 2]);
    axis image;
    axis off;
end

