% Load pre-trained VGG-16 model
net = vgg16;

% Load and preprocess the dataset (assuming 'dataset' folder contains images of people)
imds = imageDatastore('dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds.ReadFcn = @(filename)imresize(imread(filename), net.Layers(1).InputSize(1:2));

% Split the dataset into training and validation sets (80% training, 20% validation)
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% Replace the fully connected layer of the VGG-16 network with a new fully connected layer
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    net.Layers(1:end-3)
    fullyConnectedLayer(numClasses, 'Name', 'fc8', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

% Set training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 10, ...
    'ValidationPatience', Inf, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
netTransfer = trainNetwork(imdsTrain, layers, options);

% Classify images in the validation set and calculate accuracy
YPred = classify(netTransfer, imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);
disp(['Validation Accuracy: ' num2str(accuracy*100) '%']);

% Perform person recognition on new images
newImage = imread('F:\RAJESH RAJ\RP\Person Detection using deep learning\dataset\new_image.jpg');  
resizedImage = imresize(newImage, net.Layers(1).InputSize(1:2));
prediction = classify(netTransfer, resizedImage);
disp(['Predicted Label: ' char(prediction)]);

