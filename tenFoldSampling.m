clear;
clc;

% set basic info
datasetPath = 'C:\Users\IsaacLu\Desktop\meeting\資料配對\新資料\ACC_Q\';
datasetName = 'altmanAccQ_H2';

foldNumber = 5;
foldSize = 10;

% 讀取實驗資料
loadDataName = strcat(datasetPath, 'altmanAccQ_H2.mat');
load (loadDataName);

% run n-nfold 
for foldNum = 1 : foldNumber

    % get infomation of risk and normal firms
    distress = getDistressDataset(altmanAccQ_H2);
    nonDistress = getNormalDataset(altmanAccQ_H2);
    disIndex = cvpartition(size(distress,1), 'Kfold', foldSize);
    nonDisIndex = cvpartition(size(nonDistress,1), 'Kfold', foldSize);
    
    for foldS = 1 : foldSize
        coeffsLDA = []
        coeffsFTLDA = []
        coeffsOneYearPCA = []
        coeffsTwoYearPCA = []
        coeffsAllYearPCA = []
%         % prepare training data and testing data
        trainingData = [distress(disIndex.training(foldS),:); nonDistress(nonDisIndex.training(foldS),:)];
        % reduce dimension(LDA&PCA) with training data
        for i = 1:4  
%             % 1year
%             Mdl = fitcdiscr(trainingData(:, 17+(i-1)*4:20+(i-1)*4), trainingData(:, 1))
%             trainingData(:, 78+(i-1)) = trainingData(:, 17+(i-1)*4:20+(i-1)*4)*Mdl.Coeffs(1,2).Linear
%             coeffsLDA = [coeffsLDA Mdl.Coeffs(1,2).Linear]
%             %2year
%             Mdl = fitcdiscr(trainingData(:, 34+(i-1)*8:41+(i-1)*8), trainingData(:, 1))
%             trainingData(:, 88+(i-1)) = trainingData(:, 45+(i-1)*8:52+(i-1)*8)*Mdl.Coeffs(1,2).Linear
%             coeffsFTLDA = [coeffsFTLDA Mdl.Coeffs(1,2).Linear]
            %1year
            PCACoeffs1 = pca(trainingData(:, 17+(i-1)*4:20+(i-1)*4))
            trainingData(:, 67+(i-1)) = trainingData(:, 17+(i-1)*4:20+(i-1)*4)*PCACoeffs1(:,1)
            coeffsOneYearPCA = [coeffsOneYearPCA PCACoeffs1(:,1)]
              %2year
            PCACoeffs2 = pca(trainingData(:, 38+(i-1)*4:41+(i-1)*4))
            trainingData(:, 77+(i-1)) = trainingData(:, 38+(i-1)*4:41+(i-1)*4)*PCACoeffs2(:,1)
            coeffsTwoYearPCA = [coeffsTwoYearPCA PCACoeffs2(:,1)]         
            %1+2year
            PCACoeffs3 = pca(trainingData(:, 34+(i-1)*8:41+(i-1)*8))
            trainingData(:, 72+(i-1)) = trainingData(:, 34+(i-1)*8:41+(i-1)*8)*PCACoeffs3(:,1)
            coeffsAllYearPCA = [coeffsAllYearPCA PCACoeffs3(:,1)]
            
            
        end
        testingData = [distress(disIndex.test(foldS),:); nonDistress(nonDisIndex.test(foldS),:)];
        for i = 1:4
%             testingData(:, 78+(i-1)) = testingData(:, 17+(i-1)*4:20+(i-1)*4)*coeffsLDA(:, i)
            testingData(:, 67+(i-1)) = testingData(:, 17+(i-1)*4:20+(i-1)*4)*coeffsOneYearPCA(:, i)
            
            testingData(:, 77+(i-1)) = testingData(:, 38+(i-1)*4:41+(i-1)*4)*coeffsTwoYearPCA(:, i)
            
%             testingData(:, 88+(i-1)) = testingData(:, 17+(i-1)*4:20+(i-1)*4)*coeffsFTLDA(:, i)
            testingData(:, 72+(i-1)) = testingData(:, 34+(i-1)*8:41+(i-1)*8)*coeffsAllYearPCA(:, i)
        end
        distress_ = getDistressDataset(trainingData);
        nonDistress_ = getNormalDataset(trainingData);

        x = cvpartition(size(distress_,1),'kfold',foldSize);
        y = cvpartition(size(nonDistress_,1),'kfold',foldSize);
        
        for i = 1 : foldSize
        
            % preapre train and validation data
            fold.trainData{foldS}.trainingSet{i} = [distress_(x.training(i),1:end) ; nonDistress_(y.training(i),1:end)];
            fold.trainData{foldS}.validationSet{i} = [distress_(x.test(i),1:end) ; nonDistress_(y.test(i),1:end)];
%             answer = [answer ; fold.test{i}(:,1)];
        end
        
        fold.testData{foldS} = testingData;
        fold.training{foldS} = trainingData;
    end
    
    % save file
    FileName = [datasetPath, datasetName, '_', num2str(foldSize), 'fold_', num2str(foldNum), '.mat'];
    save (FileName);
end    
    % delete unused variable
%     clearvars -except foldNumber USExpDataset foldSize datasetPath datasetName;

