DATAPATH = 'C:\Users\IsaacLu\Desktop\meeting\資料配對\新資料\ACC_Q\';
DATASETNAME = 'altmanAccQ_H2_10fold';

KFOLD = 10;
FRFEATURESET = [11:15];
FRnOneYearACCURALSFEATURESET = [11:15 67:70];
FRnTwoYearACCURALSFEATURESET = [11:15 77:80];
FRnAllYearACCURALSFEATURESET = [11:15 72:75];
INFONUM = 1;
% ------------------------------------------------------
data = cell(1, 5);


%% Ensemble
%EnsembleBaggedTree
FRBagged = cell(1, 5);
FRnOneYearACCBagged = cell(1, 5);
FRnTwoYearACCBagged = cell(1, 5);
FRnAllYearACCBagged = cell(1, 5);
%% Setting 
thresholdList = linspace(0, 1, 100);
CostList = [1 1.5 2 3 5 7 7.5 8 9 10 ];
fileName = sprintf('%s_1',  DATASETNAME);

%% Initial Model
for setIter = 1:5
    fileName = sprintf('%s_%d', DATASETNAME, setIter);
    % initialized

      %% Ensemble
      %EnsembleBaggedTree
     FRBagged{setIter} = EnsembleBaggedTreeModel(KFOLD, [], thresholdList, 'M0_Bagging');
     FRnOneYearACCBagged{setIter} = EnsembleBaggedTreeModel(KFOLD, [], thresholdList, 'M1_Bagging');
     FRnTwoYearACCBagged{setIter} = EnsembleBaggedTreeModel(KFOLD, [], thresholdList, 'M2_Bagging');
     FRnAllYearACCBagged{setIter} = EnsembleBaggedTreeModel(KFOLD, [], thresholdList, 'M3_Bagging');

    %% load data
    data{setIter} = dataSet(KFOLD, DATAPATH, fileName);
    fileName = sprintf('%s_%d', DATASETNAME, setIter);
    %% Train&Test Model
    for foldIter = 1:10    
        %% Ensemble
       
        %EnsembleBaggedTree
        feature = FRFEATURESET
        FRBagged{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRBagged{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);  
        feature = FRnOneYearACCURALSFEATURESET;
        FRnOneYearACCBagged{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRnOneYearACCBagged{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
        feature = FRnTwoYearACCURALSFEATURESET;
        FRnTwoYearACCBagged{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRnTwoYearACCBagged{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
         feature = FRnAllYearACCURALSFEATURESET;
        FRnAllYearACCBagged{setIter}.trainModelandRecord(data{setIter}.gettrainingData(foldIter), feature, foldIter);
        FRnAllYearACCBagged{setIter}.testRecordedModelwithThreshold(data{setIter}.gettestData(foldIter), foldIter);
    end
end
%% Run result
test = testUnity(CostList, thresholdList, data{1}.getAllData());
