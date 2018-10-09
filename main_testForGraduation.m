DATAPATH = 'C:\Users\IsaacLu\Desktop\meeting\資料配對\新資料\ACC_Q\';
DATASETNAME = 'altmanAccQ_H2_10fold';

KFOLD = 10;
FRFEATURESET = [11:15];
FRnACCURALSFEATURESET = [11:15 17:32];%+first year period acc
FRnFTACCURALSFEATURESET = [11:15 38:41 46:49 54:57 62:65];%+second year period acc
FRnAllYearACCURALSFEATURESET = [11:15 34:65];%+all year period acc
INFONUM = 1;
% ------------------------------------------------------
data = cell(1, 5);

%% SVM
%LSVM
FRLSVM = cell(1, 5);
FRACCLSVM = cell(1, 5)
FRFTACCLSVM = cell(1, 5)
FRAllACCLSVM = cell(1, 5);
% %QSVM
% FRQSVM = cell(1, 5);
% FRACCQSVM = cell(1, 5)
% FRFTACCQSVM = cell(1, 5)
% %MediumRBFSVM
% FRMediumRBFSVM = cell(1, 5);
% FRACCMediumRBFSVM = cell(1, 5)
% FRFTACCMediumRBFSVM = cell(1, 5)
%% D.Tree
% %MediumTree
% FRMediumTree = cell(1, 5);
% FRACCMediumTree = cell(1, 5)
% FRFTACCMediumTree = cell(1, 5)
%CART
FRCART = cell(1, 5);
FRACCCART = cell(1, 5)
FRFTACCCART = cell(1, 5)
FRAllACCCART = cell(1, 5)
%% LDA
%LDA
FRLDA = cell(1, 5);
FRACCLDA = cell(1, 5)
FRFTACCLDA = cell(1, 5)
FRAllACCLDA = cell(1, 5)
%% Bayes
FRBayes = cell(1, 5);
FRACCBayes = cell(1, 5)
FRFTACCBayes = cell(1, 5)
FRAllACCBayes = cell(1, 5)
%% KNN
%MediumKNN
FRMKNN = cell(1, 5);
FRACCMKNN = cell(1, 5)
FRFTACCMKNN = cell(1, 5)
FRAllACCMKNN = cell(1, 5)
% %CosineKNN
% FRCKNN = cell(1, 5);
% FRACCCKNN = cell(1, 5)
% FRFTACCCKNN = cell(1, 5)
% %WeightedKNN
% FRWKNN = cell(1, 5);
% FRACCWKNN = cell(1, 5)
% FRFTACCWKNN = cell(1, 5)

%% Ensemble
%EnsembleBoostedTree
FRBoosted = cell(1, 5);
FRACCBoosted = cell(1, 5)
FRFTACCBoosted = cell(1, 5)
FRAllACCBoosted = cell(1, 5)
%EnsembleBaggedTree
FRBagged = cell(1, 5);
FRACCBagged = cell(1, 5)
FRFTACCBagged = cell(1, 5)
FRAllACCBagged = cell(1, 5)
%EnsembleBaggedSVM
FRBaggedSVM = cell(1, 5);
FRACCBaggedSVM = cell(1, 5)
FRFTACCBaggedSVM = cell(1, 5)
FRAllACCBaggedSVM = cell(1, 5)
%Random forest
% FRRF = cell(1, 5);
% FRACCRF = cell(1, 5)
% FRFTACCRF = cell(1, 5)
% FRAllACCRF = cell(1, 5)
%% Setting 
thresholdList = linspace(0, 1, 100);
CostList = [1 1.5 2 3 5 7 7.5 8 9 10 ];
fileName = sprintf('%s_1',  DATASETNAME);

%% Initial Model
for setIter = 1:5
    fileName = sprintf('%s_%d', DATASETNAME, setIter);
    % initialized

%     ttest{setIter} = tTestFS(KFOLD, FRFEATURESET, INFONUM);
%% SVM
%LSVM
     FRLSVM{setIter} = SVMModel(KFOLD, [], thresholdList, 'M0_LSVM');
     FRACCLSVM{setIter} = SVMModel(KFOLD, [], thresholdList, 'M1_LSVM');
     FRFTACCLSVM{setIter} = SVMModel(KFOLD, [], thresholdList, 'M2_LSVM');
     FRAllACCLSVM{setIter} = SVMModel(KFOLD, [], thresholdList, 'M3_LSVM');
%     %QSVM
%      FRQSVM{setIter} = QSVMModel(KFOLD, [], thresholdList, 'M0');
%      FRACCQSVM{setIter} = QSVMModel(KFOLD, [], thresholdList, 'M1');
%      FRFTACCQSVM{setIter} = QSVMModel(KFOLD, [], thresholdList, 'M2'); 
%      %MediumRBFSVM
%      FRMediumRBFSVM{setIter} = MediumRBFSVMModel(KFOLD, [], thresholdList, 'M0');
%      FRACCMediumRBFSVM{setIter} = MediumRBFSVMModel(KFOLD, [], thresholdList, 'M1');
%      FRFTACCMediumRBFSVM{setIter} = MediumRBFSVMModel(KFOLD, [], thresholdList, 'M2');
 %% D.Tree
 %MediumTree  
%      FRMediumTree{setIter} = MediumTreeModel(KFOLD, [], thresholdList, 'M0');
%      FRACCMediumTree{setIter} = MediumTreeModel(KFOLD, [], thresholdList, 'M1');
%      FRFTACCMediumTree{setIter} = MediumTreeModel(KFOLD, [], thresholdList, 'M2');
     %CART
     FRCART{setIter} = CARTModel(KFOLD, [], thresholdList, 'M0_CART');
     FRACCCART{setIter} = CARTModel(KFOLD, [], thresholdList, 'M1_CART');
     FRFTACCCART{setIter} = CARTModel(KFOLD, [], thresholdList, 'M2_CART');
     FRAllACCCART{setIter} = CARTModel(KFOLD, [], thresholdList, 'M3_CART');
     %% LDA
 %LDA  
     FRLDA{setIter} = LDAModel(KFOLD, [], thresholdList, 'M0_LDA');
     FRACCLDA{setIter} = LDAModel(KFOLD, [], thresholdList, 'M1_LDA');
     FRFTACCLDA{setIter} = LDAModel(KFOLD, [], thresholdList, 'M2_LDA');
     FRAllACCLDA{setIter} = LDAModel(KFOLD, [], thresholdList, 'M3_LDA');
     %% Bayes
     %bayes
     FRBayes{setIter} = BayesModel(KFOLD, [], thresholdList, 'M0_Bayes');
     FRACCBayes{setIter} = BayesModel(KFOLD, [], thresholdList, 'M1_Bayes');
     FRFTACCBayes{setIter} = BayesModel(KFOLD, [], thresholdList, 'M2_Bayes');
     FRAllACCBayes{setIter} = BayesModel(KFOLD, [], thresholdList, 'M3_Bayes');
          %% KNN
 %Mediun KNN
     FRMKNN{setIter} = MediumKNNModel(KFOLD, [], thresholdList, 'M0_KNN');
     FRACCMKNN{setIter} = MediumKNNModel(KFOLD, [], thresholdList, 'M1_KNN');
     FRFTACCMKNN{setIter} = MediumKNNModel(KFOLD, [], thresholdList, 'M2_KNN');
     FRAllACCMKNN{setIter} = MediumKNNModel(KFOLD, [], thresholdList, 'M3_KNN');
      %Cosine KNN
%      FRCKNN{setIter} = CosineKNNModel(KFOLD, [], thresholdList, 'M0');
%      FRACCCKNN{setIter} = CosineKNNModel(KFOLD, [], thresholdList, 'M1');
%      FRFTACCCKNN{setIter} = CosineKNNModel(KFOLD, [], thresholdList, 'M2');
%      %Weighted KNN
%      FRWKNN{setIter} = WeightedKNNModel(KFOLD, [], thresholdList, 'M0');
%      FRACCWKNN{setIter} = WeightedKNNModel(KFOLD, [], thresholdList, 'M1');
%      FRFTACCWKNN{setIter} = WeightedKNNModel(KFOLD, [], thresholdList, 'M2');
      %% Ensemble
 %EnsembleBoostedTree
     FRBoosted{setIter} = EnsembleBoostedTreeModel(KFOLD, [], thresholdList, 'M0_Boosting');
     FRACCBoosted{setIter} = EnsembleBoostedTreeModel(KFOLD, [], thresholdList, 'M1_Boosting');
     FRFTACCBoosted{setIter} = EnsembleBoostedTreeModel(KFOLD, [], thresholdList, 'M2_Boosting');
     FRAllACCBoosted{setIter} = EnsembleBoostedTreeModel(KFOLD, [], thresholdList, 'M3_Boosting');
      %EnsembleBaggedTree
     FRBagged{setIter} = EnsembleBaggedTreeModel(KFOLD, [], thresholdList, 'M0_Bagging');
     FRACCBagged{setIter} = EnsembleBaggedTreeModel(KFOLD, [], thresholdList, 'M1_Bagging');
     FRFTACCBagged{setIter} = EnsembleBaggedTreeModel(KFOLD, [], thresholdList, 'M2_Bagging');
     FRAllACCBagged{setIter} = EnsembleBaggedTreeModel(KFOLD, [], thresholdList, 'M3_Bagging');
      %EnsembleBaggedSVM
     FRBaggedSVM{setIter} = EnsembleBaggedSVMModel(KFOLD, [], thresholdList, 'M0_BaggingSVM');
     FRACCBaggedSVM{setIter} = EnsembleBaggedSVMModel(KFOLD, [], thresholdList, 'M1_BaggingSVM');
     FRFTACCBaggedSVM{setIter} = EnsembleBaggedSVMModel(KFOLD, [], thresholdList, 'M2_BaggingSVM');
     FRAllACCBaggedSVM{setIter} = EnsembleBaggedSVMModel(KFOLD, [], thresholdList, 'M3_BaggingSVM');
     %Random Forest
%      FRRF{setIter} = RandomForestModel(KFOLD, [], thresholdList, 'M0_RF');
%      FRACCRF{setIter} = RandomForestModel(KFOLD, [], thresholdList, 'M1_RF');
%      FRFTACCRF{setIter} = RandomForestModel(KFOLD, [], thresholdList, 'M2_RF');
%      FRAllACCRF{setIter} = RandomForestModel(KFOLD, [], thresholdList, 'M3_RF');
    %% load data
    data{setIter} = dataSet(KFOLD, DATAPATH, fileName);
    fileName = sprintf('%s_%d', DATASETNAME, setIter);
    %% Train&Test Model
    for foldIter = 1:10    
        %% SVM
        %LSVM
        feature = FRFEATURESET
        FRLSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRLSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
        feature = FRnACCURALSFEATURESET;
        FRACCLSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRACCLSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnFTACCURALSFEATURESET;
        FRFTACCLSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRFTACCLSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnAllYearACCURALSFEATURESET;
        FRAllACCLSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRAllACCLSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
%         %QSVM
%         feature = FRFEATURESET
%         FRQSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRQSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
%         feature = FRnACCURALSFEATURESET;
%         FRACCQSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRACCQSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
%         feature = FRnFTACCURALSFEATURESET;
%         FRFTACCQSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRFTACCQSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
%         %RBFSVM
%         feature = FRFEATURESET
%         FRMediumRBFSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRMediumRBFSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
%         feature = FRnACCURALSFEATURESET;
%         FRACCMediumRBFSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRACCMediumRBFSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
%         feature = FRnFTACCURALSFEATURESET;
%         FRFTACCMediumRBFSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRFTACCMediumRBFSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        
        %% Ｄ. Tree
%         %MediumTree
%         feature = FRFEATURESET
%         FRMediumTree{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRMediumTree{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
%         feature = FRnACCURALSFEATURESET;
%         FRACCMediumTree{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRACCMediumTree{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
%         feature = FRnFTACCURALSFEATURESET;
%         FRFTACCMediumTree{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRFTACCMediumTree{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
       %CART
        feature = FRFEATURESET
        FRCART{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRCART{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
        feature = FRnACCURALSFEATURESET;
        FRACCCART{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRACCCART{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnFTACCURALSFEATURESET;
        FRFTACCCART{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRFTACCCART{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnAllYearACCURALSFEATURESET;
        FRAllACCCART{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRAllACCCART{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        %% LDA
        %LDA
        feature = FRFEATURESET
        FRLDA{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRLDA{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
        feature = FRnACCURALSFEATURESET;
        FRACCLDA{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRACCLDA{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnFTACCURALSFEATURESET;
        FRFTACCLDA{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRFTACCLDA{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
         feature = FRnAllYearACCURALSFEATURESET;
        FRAllACCLDA{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRAllACCLDA{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
       %% bayes
        feature = FRFEATURESET
        FRBayes{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRBayes{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
        feature = FRnACCURALSFEATURESET;
        FRACCBayes{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRACCBayes{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnFTACCURALSFEATURESET;
        FRFTACCBayes{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRFTACCBayes{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnAllYearACCURALSFEATURESET;
        FRAllACCBayes{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRAllACCBayes{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        %% KNN
        %Medium KNN
        feature = FRFEATURESET
        FRMKNN{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRMKNN{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
        feature = FRnACCURALSFEATURESET;
        FRACCMKNN{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRACCMKNN{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnFTACCURALSFEATURESET;
        FRFTACCMKNN{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRFTACCMKNN{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
         feature = FRnAllYearACCURALSFEATURESET;
        FRAllACCMKNN{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRAllACCMKNN{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        %Cosine KNN
%         feature = FRFEATURESET
%         FRCKNN{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRCKNN{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
%         feature = FRnACCURALSFEATURESET;
%         FRACCCKNN{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRACCCKNN{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
%         feature = FRnFTACCURALSFEATURESET;
%         FRFTACCCKNN{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRFTACCCKNN{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
%         %Weighted KNN
%         feature = FRFEATURESET
%         FRWKNN{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRWKNN{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
%         feature = FRnACCURALSFEATURESET;
%         FRACCWKNN{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRACCWKNN{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
%         feature = FRnFTACCURALSFEATURESET;
%         FRFTACCWKNN{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRFTACCWKNN{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        %% Ensemble
        %EnsembleBoostedTree
        feature = FRFEATURESET
        FRBoosted{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRBoosted{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
        feature = FRnACCURALSFEATURESET;
        FRACCBoosted{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRACCBoosted{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnFTACCURALSFEATURESET;
        FRFTACCBoosted{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRFTACCBoosted{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
         feature = FRnAllYearACCURALSFEATURESET;
        FRAllACCBoosted{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRAllACCBoosted{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        %EnsembleBaggedTree
        feature = FRFEATURESET
        FRBagged{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRBagged{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
        feature = FRnACCURALSFEATURESET;
        FRACCBagged{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRACCBagged{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnFTACCURALSFEATURESET;
        FRFTACCBagged{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRFTACCBagged{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnAllYearACCURALSFEATURESET;
        FRAllACCBagged{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRAllACCBagged{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        %EnsembleBaggedSVM
        feature = FRFEATURESET
        FRBaggedSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRBaggedSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
        feature = FRnACCURALSFEATURESET;
        FRACCBaggedSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRACCBaggedSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnFTACCURALSFEATURESET;
        FRFTACCBaggedSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRFTACCBaggedSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnAllYearACCURALSFEATURESET;
        FRAllACCBaggedSVM{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRAllACCBaggedSVM{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        %RandomForest
        
%         feature = FRFEATURESET
%         FRRF{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRRF{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
%         feature = FRnACCURALSFEATURESET;
%         FRACCRF{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRACCRF{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
%         feature = FRnFTACCURALSFEATURESET;
%         FRFTACCRF{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRFTACCRF{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
%         feature = FRnAllYearACCURALSFEATURESET;
%         FRAllACCRF{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
%         FRAllACCRF{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
    end
end
%% Run result
test = testUnity(CostList, thresholdList, data{1}.getAllData());
