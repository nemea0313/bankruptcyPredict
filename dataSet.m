classdef dataSet < handle
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = public )
        PATH;
        FILENAME;
        KFOLD;
        kFoldDataset;
        trainingData;
        testData;
        % trainingData and validationData is 10 fold of trainData
        % use for derive training dataset for metadata (stacking)
        % and prevent information leak
        trainData;
        validationData;
        boostrapNum;
    end
    
    methods
        function obj = dataSet(kfold,path, boostrapNum , fileName)
            obj.KFOLD = kfold;
            obj.boostrapNum = boostrapNum;
            obj.setPATH(path);
            obj.setFILENAME(fileName);
            obj.loadData();
        end
        %% get/set function
        function setFILENAME(obj,fileName)
            obj.FILENAME = fileName;
        end
        function setPATH(obj, path)
            obj.PATH = path;
        end
        function loadData(obj)
            loadDataName = sprintf('%s/%s', obj.PATH, obj.FILENAME);
            load(loadDataName, 'fold');
            obj.setkFoldDataset(fold);
            obj.settrainingData(fold.training);
            obj.settestData(fold.testData);
            obj.setvalidationDataSet(fold);
        end
        function setkFoldDataset(obj, fold)
            obj.kFoldDataset = fold;
        end
        function settrainingData(obj, trainingData)
            obj.trainingData = trainingData;
        end
        function output=gettrainingData(obj, curFold)
            output = obj.trainingData{curFold};
        end
        function settestData(obj, testData)
            obj.testData = testData;
        end
        function output=gettestData(obj, curFold)
            output = obj.testData{curFold};
        end
        function setvalidationDataSet(obj,fold)
            obj.trainData = cell(1, obj.KFOLD);
            obj.validationData = cell(1,obj.KFOLD);
            for i = 1:obj.KFOLD
                obj.trainData{i} = fold.trainData{i}.trainingSet;
                obj.validationData{i} = fold.trainData{i}.validationSet;
            end
        end
        function output = gettrainData(obj, curFold, subFold)
            output = obj.trainData{curFold}{subFold};
        end
        function output = getvalidationData(obj, curFold, subFold)
            output = obj.validationData{curFold}{subFold};
        end
        function output = getAllData(obj)
            output = obj.kFoldDataset;
        end
        function output = getTrainingDistressedFirmsData(obj, curFold)
            distressedFirmsIndex = obj.trainingData{curFold}(:, 1) == 1;
            output = obj.trainingData{curFold}(distressedFirmsIndex, :);
        end
          function output = getTrainingNormalFirmsData(obj, curFold)
            normalFirmsIndex = obj.trainingData{curFold}(:, 1) == 0;
            output = obj.trainingData{curFold}(normalFirmsIndex, :);
        end 
    end
end
