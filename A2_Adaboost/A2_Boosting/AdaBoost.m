%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 25;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 500;   
% Number of weak classifiers
nbrWeakClassifiers = 30;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

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
n = nbrTrainImages;

d = ones(1, n )/n; % inital weight
M = 6; %number of classifier


pm = zeros(1,M);
tm = zeros(1,M);
e_min = inf;
am = zeros(1,M);
idm = zeros(1,M);

%iter the classifier
for m = 1:M 
    
    %iter the features
    for f = 1:size(xTrain,1) 
        
        %iter the thresholds
        for t= 1:size(xTrain,2)
            
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
            
            if E < e_min
                e_min = E;
                best_p = p;
                best_t = threshold;
                best_id = f;
                best_c = C;
           
            end
            
        end
    end
    
    pm(m) = best_p;
    tm(m) = best_t;
    idm(m) = best_id;
        
    
    %update weight
    am(m) = 0.5*log((1-e_min)/e_min );
    d = d.* exp(-a(m)* (yTrain.*best_c));
    d = d/sum(d);
end


                
          
%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

classifier = zeros(m ,size(yTest,2));
for i = 1:m
    classifier(i,:) = am(i).*WeakClassifier(tm(i), pm(i), xTest(idm(i),:)  );
end

strongClassifier = sign(sum(classifier));
    



    
%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.


%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.


