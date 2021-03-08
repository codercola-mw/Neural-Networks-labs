%% This script will help you test your multi-layer neural network code
run setupSupervisedLab.m
%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 3; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );
%plotCase(X,D)
%% Select a subset of the training features

numBins = 2;                    % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select features at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);

% Add your own code to setup data for training and test here
 XTrain = XBins{1}
 DTrain = DBins{1}
 LTrain = LBins{1}
 XTest  = XBins{2}
 DTest  = DBins{2}
 LTest  = LBins{2}

%% Modify the X Matrices so that a bias is added
%  Note that the bias must be the last feature for the plot code to work

% The training data
XTrain = [XTrain ones(length(XTrain),1)]

% The test data
XTest = [XTest ones(length(XTest),1)]

%% Train your multi-layer network
%  Note: You need to modify trainMultiLayer() and runMultiLayer()
%  in order to train the network

numHidden = 8;  %20 0.94  % use 35
numIterations = 10000;  % 4: 30000
learningRate = 0.05;  %0.0009
%0.002 0.74 %0.0002 0.44  %0.0009 0.67 0.94 %0.001 0.93

W0 = randn([size(XTrain, 2) numHidden])/10;
V0 = randn([numHidden+1 size(DTrain, 2)])/10; %change to [4 3 ]

% Run training loop
tic;
[W,V,ErrTrain,ErrTest] = trainMultiLayer(XTrain, DTrain, XTest, DTest ,W0, V0, numIterations, learningRate);
trainingTime = toc;

%% Plot errors
%  Note: You should not have to modify this code

[minErrTest, minErrTestInd] = min(ErrTest);

figure(1101);
clf;
semilogy(ErrTrain, 'k', 'linewidth', 1.5);
hold on;
semilogy(ErrTest, 'r', 'linewidth', 1.5);
semilogy(minErrTestInd, minErrTest, 'bo', 'linewidth', 1.5);
hold off;
xlim([0,numIterations]);
grid on;
title('Training and Test Errors, Multi-layer');
legend('Training Error', 'Test Error', 'Min Test Error');
xlabel('Epochs');
ylabel('Error');

%% Calculate the Confusion Matrix and the Accuracy of the data
%  Note: you have to modify the calcConfusionMatrix() and calcAccuracy()
%  functions yourself.

tic;
[~, LPredTrain] = runMultiLayer(XTrain, W, V);
[~, LPredTest ] = runMultiLayer(XTest , W, V);
classificationTime = toc/(length(XTest) + length(XTrain));

% The confucionMatrix
cM = calcConfusionMatrix(LPredTest, LTest);
cM2 = calcConfusionMatrix(LPredTrain,LTrain);

% The accuracy
acc = calcAccuracy(cM);
acc2 = calcAccuracy(cM2);


disp(['Time spent training: ' num2str(trainingTime) ' sec']);
disp(['Time spent classifying 1 sample: ' num2str(classificationTime) ' sec']);
disp(['Test accuracy: ' num2str(acc)]);
disp(['Train accuracy: ' num2str(acc2)]);
%% Plot classifications
%  Note: You should not have to modify this code

if dataSetNr < 4
    plotResultDots(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, 'multi', {W,V}, []);
else
    plotResultsOCR(XTest, LTest, LPredTest);
end
%% non - generalizable 
% use a small set of data to train a the rest for testing

% split the data into 1/30
numBins = 100;
 [Xs, Ds, Ls] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );
 
XTrain = Xs{1};
DTrain = Ds{1};
LTrain = Ls{1};
XTest  = combineBins(Xs(2:numBins), 1:numBins-1);
DTest  = combineBins(Ds(2:numBins), 1:numBins-1);
LTest  = combineBins(Ls(2:numBins), 1:numBins-1);
 
