function fmeaavg = speech_classifier(datasetFolder, labelsfile, samplingfreq, priornet, outputnet)

NumAugmentations = 1;
adsAug = audioDatastore(datasetFolder);
adsAug.Files = natsortfiles(adsAug.Files)

labelTable = readtable(labelsFile);
labelTable.Emotion = categorical(labelTable.Emotion);
labelTable.Speaker = categorical(labelTable.Speaker);

fs = samplingfreq; 
adsAug.Labels = labelTable;

win = hamming(round(0.03*fs),"periodic");
overlapLength = 0;

afe = audioFeatureExtractor( ...
    'Window',win, ...
    'OverlapLength',overlapLength, ...
    'SampleRate',fs, ...
    ...
    'gtcc',true, ...
    'gtccDelta',true, ...
    'mfccDelta',true, ...
    ...
    'SpectralDescriptorInput','melSpectrum', ...
    'spectralCrest',true);

adsTrain = adsAug;

tallTrain = tall(adsTrain);

featuresTallTrain = cellfun(@(x)extract(afe,x),tallTrain,"UniformOutput",false);
featuresTallTrain = cellfun(@(x)pagectranspose(x),featuresTallTrain,"UniformOutput",false);
featuresTrain = gather(featuresTallTrain);

mins = 10000;
for l1=1:size(featuresTrain,1)
   if size(featuresTrain{l1},2) < mins
       mins = size(featuresTrain{l1},2);
   end
end

for i = 1:size(featuresTrain,1)
    featuresTrain{i,1} = featuresTrain{i,1}(:,1:mins); 
end
allFeatures = cat(2,featuresTrain{:});
M = mean(allFeatures,2,'omitnan');
S = std(allFeatures,0,2,'omitnan');

featuresTrain = cellfun(@(x)(x-M)./S,featuresTrain,'UniformOutput',false);

featureVectorsPerSequence = 5;
featureVectorOverlap = 3;
[sequencesTrain,sequencePerFileTrain] = HelperFeatureVector2Sequence(featuresTrain,featureVectorsPerSequence,featureVectorOverlap);

augads = adsAug;
ads = adsAug;
extractor = afe;

speaker = categories(ads.Labels.Speaker);
emptyEmotions = (ads.Labels.Emotion);
emptyEmotions(:) = [];

% Loop over each fold.
trueLabelsCrossFold = {};
predictedLabelsCrossFold = {};
numFolds = numel(speaker);
for i = 1:numFolds
        i
        % 1. Divide the audio datastore into training and validation sets.
        % Convert the data to tall arrays.
        idxTrain           = augads.Labels.Speaker~=speaker(i);
        augadsTrain        = subset(augads,idxTrain);
        augadsTrain.Labels = augadsTrain.Labels.Emotion;
        tallTrain          = tall(augadsTrain);
        idxValidation        = ads.Labels.Speaker==speaker(i);
        adsValidation        = subset(ads,idxValidation);
        adsValidation.Labels = adsValidation.Labels.Emotion;
        tallValidation       = tall(adsValidation);

        % 2. Extract features from the training set. Reorient the features
        % so that time is along rows to be compatible with
        % sequenceInputLayer.
        tallTrain         = cellfun(@(x)x/max(abs(x),[],'all'),tallTrain,"UniformOutput",false);
        tallFeaturesTrain = cellfun(@(x)extract(extractor,x),tallTrain,"UniformOutput",false);
        tallFeaturesTrain = cellfun(@(x)pagectranspose(x),tallFeaturesTrain,"UniformOutput",false);  %#ok<NASGU>
        [~,featuresTrain] = evalc('gather(tallFeaturesTrain)'); % Use evalc to suppress command-line output.
        tallValidation         = cellfun(@(x)x/max(abs(x),[],'all'),tallValidation,"UniformOutput",false);
        tallFeaturesValidation = cellfun(@(x)extract(extractor,x),tallValidation,"UniformOutput",false);
        tallFeaturesValidation = cellfun(@(x)pagectranspose(x),tallFeaturesValidation,"UniformOutput",false); %#ok<NASGU>
        [~,featuresValidation] = evalc('gather(tallFeaturesValidation)'); % Use evalc to suppress command-line output.

        % 3. Use the training set to determine the mean and standard
        % deviation of each feature. Normalize the training and validation
        % sets.
        mins = 10000;
        for l1=1:size(featuresTrain,1)
          if size(featuresTrain{l1},2) < mins
            mins = size(featuresTrain{l1},2);
          end
        end
        for ife = 1:size(featuresTrain,1)
                featuresTrain{ife,1} = featuresTrain{ife,1}(:,1:mins); 
        end
        allFeatures = cat(2,featuresTrain{:});
        M = mean(allFeatures,2,'omitnan');
        S = std(allFeatures,0,2,'omitnan');
        featuresTrain = cellfun(@(x)(x-M)./S,featuresTrain,'UniformOutput',false);
        for ii = 1:numel(featuresTrain)
            idx = find(isnan(featuresTrain{ii}));
            if ~isempty(idx)
                featuresTrain{ii}(idx) = 0;
            end
        end
        mins = 10000;
        for l1=1:size(featuresValidation,1)
          if size(featuresValidation{l1},2) < mins
            mins = size(featuresValidation{l1},2);
          end
        end
        for ife = 1:size(featuresValidation,1)
                featuresValidation{ife,1} = featuresValidation{ife,1}(:,1:mins); 
        end
        
        featuresValidation = cellfun(@(x)(x-M)./S,featuresValidation,'UniformOutput',false);
        for ii = 1:numel(featuresValidation)
            idx = find(isnan(featuresValidation{ii}));
            if ~isempty(idx)
                featuresValidation{ii}(idx) = 0;
            end
        end

        % 4. Buffer the sequences so that each sequence consists of twenty
        % feature vectors with overlaps of 10 feature vectors.
        featureVectorsPerSequence = 5;
        featureVectorOverlap = 2;
        [sequencesTrain,sequencePerFileTrain] = HelperFeatureVector2Sequence(featuresTrain,featureVectorsPerSequence,featureVectorOverlap);
        [sequencesValidation,sequencePerFileValidation] = HelperFeatureVector2Sequence(featuresValidation,featureVectorsPerSequence,featureVectorOverlap);

        % 5. Replicate the labels of the train and validation sets so that
        % they are in one-to-one correspondence with the sequences.
        labelsTrain = [emptyEmotions;augadsTrain.Labels];
        labelsTrain = labelsTrain(:);
        labelsTrain = repelem(labelsTrain,[sequencePerFileTrain{:}]);

        % 6. Define a BiLSTM network.
        dropoutProb1 = 0.3;
        numUnits     = 5;
        dropoutProb2 = 0.3;
        layers = [ ...
            sequenceInputLayer(size(sequencesTrain{1},1))
            %dropoutLayer(dropoutProb1)
            %bilstmLayer(numUnits,"OutputMode","last")
            dropoutLayer(dropoutProb1)
            bilstmLayer(numUnits,"OutputMode","last")
            dropoutLayer(dropoutProb2)
            fullyConnectedLayer(numel(categories(emptyEmotions)))
            softmaxLayer
            classificationLayer];

        % 7. Define training options.
        miniBatchSize       = 500;
        initialLearnRate    = 0.005;
        learnRateDropPeriod = 2;
        maxEpochs           = 5;
        options = trainingOptions("adam", ...
            "MiniBatchSize",miniBatchSize, ...
            "InitialLearnRate",initialLearnRate, ...
            "LearnRateDropPeriod",learnRateDropPeriod, ...
            "LearnRateSchedule","piecewise", ...
            "MaxEpochs",maxEpochs, ...
            "Shuffle","every-epoch", ...
            "Verbose",false);
           % "Plots","Training-Progress");

        % 8. Train the network.
        neta = load(priornet);
        net = trainNetwork(sequencesTrain,labelsTrain,neta.net.Layers,options)
       
        % 9. Evaluate the network. Call classify to get the predicted labels
        % for each sequence. Get the mode of the predicted labels of each
        % sequence to get the predicted labels of each file.
        predictedLabelsPerSequence = classify(net,sequencesValidation);
        trueLabels = categorical(adsValidation.Labels);
        predictedLabels = trueLabels;
        idx1 = 1;
        for ii = 1:numel(trueLabels)
            predictedLabels(ii,:) = mode(predictedLabelsPerSequence(idx1:idx1 + sequencePerFileValidation{ii} - 1,:),1);
            idx1 = idx1 + sequencePerFileValidation{ii};
        end
        trueLabelsCrossFold{i} = trueLabels; %#ok<AGROW>
        predictedLabelsCrossFold{i} = predictedLabels; %#ok<AGROW>
end

labelsTrue = trueLabelsCrossFold;
labelsPred = predictedLabelsCrossFold;

for ii = 1:numel(labelsTrue)
    foldAcc = mean(labelsTrue{ii}==labelsPred{ii})*100;
    fprintf('Fold %1.0f, Accuracy = %0.1f\n',ii,foldAcc);
end

labelsTrueMat = cat(1,labelsTrue{:});
labelsPredMat = cat(1,labelsPred{:});
figure
cm = confusionchart(labelsTrueMat,labelsPredMat);
valAccuracy = mean(labelsTrueMat==labelsPredMat)*100;
cm.Title = sprintf('Confusion Matrix for 10-Fold Cross-Validation\nAverage Accuracy = %0.1f',valAccuracy);
sortClasses(cm,categories(emptyEmotions))
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

cm2 = confusionmat(labelsTrueMat,labelsPredMat);
nclass = 3;

% Calculate F-measure
for x=1:nclass

tp = cm2(x,x);
tn = cm2(1,1);
for y=2:nclass
tn = tn+cm2(y,y);
end
tn = tn-cm2(x,x);

fp = sum(cm2(:, x))-cm2(x, x);
fn = sum(cm2(x, :), 2)-cm2(x, x);
pre(x)=tp/(tp+fp+0.01);
rec(x)=tp/(tp+fn+0.01);
fmea(x) = 2*pre(x)*rec(x)/(pre(x)+rec(x)+0.01);
end

fmeaavg = mean(fmea);
dlmwrite('fmeasure.txt',fmea);

save(outputnet,'net');
end