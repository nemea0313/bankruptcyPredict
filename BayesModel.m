classdef BayesModel < handle & modelToolkit
    %UNTITLED12 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        KFOLD;
        DISCRIPTION;
        model;
        testingResult;
        trainingError;
        ConfusionMatrix;
        penaltyList;
        prob;
        thresholdList;
        FeatureSet;
        FeatureWeightSet;
    end

    methods
        function obj = BayesModel(kfold, penaltyList,  thresholdList, discription)
            obj.KFOLD = kfold;
            obj.penaltyList = penaltyList;
            obj.prob = cell(1, kfold);
            obj.testingResult = cell(1, kfold);
            obj.thresholdList = thresholdList;
            obj.DISCRIPTION = discription;
            obj.FeatureSet = cell(1, obj.KFOLD);
            obj.FeatureWeightSet = cell(1, obj.KFOLD);
            if isempty(penaltyList)
                obj.ConfusionMatrix = zeros(3, length(thresholdList)*obj.KFOLD);
                obj.trainingError = zeros(3, obj.KFOLD);

                obj.model = cell(1, obj.KFOLD);
            else
                obj.ConfusionMatrix = zeros(3, length(penaltyList)*obj.KFOLD);
                obj.trainingError = zeros(3, length(penaltyList)*obj.KFOLD);
                obj.model = cell(1, KFOLD * length(penaltyList));
            end
        end
        function model = trainModelandRecord(obj, trainData, FeatureSet, curFold)

            model = fitcnb(trainData(:, FeatureSet), trainData(:,1));
            obj.model{curFold} = model;
            obj.FeatureSet{curFold} = FeatureSet;
        end
        function testRecordedModel(obj, testData, curFold)

            Featureset = obj.FeatureSet{curFold};
            Model = obj.model{curFold};
            [result, probResult, ~] = predict(Model, testData(:, Featureset));
            obj.testingResult{curFold} = result;
            obj.prob{curFold} = probResult;
            obj.ConfusionMatrix(:, curFold) = obj.getConfusionMatrix(testData(:, 1), result);
        end
        function model = trainModel(obj, trainData, FeatureSet)

            model =  fitcnb(trainData(:, FeatureSet), trainData(:,1));
        end
        function [confusionMatrix, result, probResult ]= testModel(obj, testData, Featureset, Model)
            [result, probResult, ~] = predict(Model, testData(:, Featureset));
            confusionMatrix=obj.getConfusionMatrix(testData(:, 1), result);
        end
        function output = getConfusionMatrix(obj, testingAns, testingResult)
            correctCount = 0; %�P�_���`���`��
            typeIICount = 0;     %���׬O�M���Q�~�P�����`�`��
            typeICount = 0;     %���׬O���`�Q�~�P���M���`��
            numOfTestingData = numel(testingAns); %���X��testingData
            numOfNormalAns = 0; %���`���צ��X��
            numOfDistressAns = 0; %�M�����צ��X��
            %�V�m�����p��FAR&FRR����
            for v = 1:numOfTestingData

                if ( testingAns(v) == 0 ) %���`����
                    numOfNormalAns = numOfNormalAns + 1;
                else % �M������
                    numOfDistressAns = numOfDistressAns + 1;
                end

                if ( testingAns(v) == 0 && testingResult(v) == 1 ) %���`�P���M��
                    typeIICount = typeIICount + 1;
                elseif ( testingAns(v) == 1 && testingResult(v) == 0 ) %�M���P�����`
                    typeICount = typeICount + 1;
                else %�P�_���`
                    correctCount = correctCount + 1;
                end
            end
            accuracy = correctCount / numOfTestingData;
            typeI = typeICount / numOfDistressAns;
            typeII = typeIICount / numOfNormalAns;
            output = [accuracy; typeI; typeII];
        end
        function [result, probResult]=testRecordedModelwithThreshold(obj, testData, curFold)
            Featureset = obj.FeatureSet{curFold};

            Model = obj.model{curFold};
            curTime = (curFold-1)*length(obj.thresholdList)+1;
            [result, probResult, ~] = predict(Model, testData(:, Featureset));
            for piter = 1:length(obj.thresholdList)
                threshold = obj.thresholdList(piter);
                resultT = [];
                for titer = 1:size(probResult)

                    if probResult(titer, 2) >= threshold
                        resultT(titer, 1) = 1;
                    else
                        resultT(titer, 1) = 0;
                    end
                end
                obj.testingResult{curFold}=[obj.testingResult{curFold} resultT];
                obj.prob{curFold} = probResult;
                obj.ConfusionMatrix(:, curTime) = obj.getConfusionMatrix(testData(:, 1), resultT);
                curTime = curTime + 1;
            end

        end
        function [confusionMatrix, resultList, probResult ]= testModelwithThreshold(obj, testData, Featureset, Model)
            [result, probResult, ~] = predict(Model, testData(:, Featureset));
            for piter = 1:length(obj.thresholdList)
                threshold = obj.thresholdList(piter);
                resultT = [];
                for titer = 1:size(probResult)

                    if probResult(titer, 2) >= threshold
                        resultT(titer, 1) = 1;
                    else
                        resultT(titer, 1) = 0;
                    end
                end
                resultList{piter} = resultT;
                confusionMatrix(:, piter) = obj.getConfusionMatrix(testData(:, 1), resultT);
            end
%             confusionMatrix=obj.getConfusionMatrix(testData(:, 1), resultT);
        end
        function [output] = getConfusionMatrixList(obj)
            output = obj.ConfusionMatrix;
        end
        function output = getFeatureSet(obj, curFold)
            output = obj.FeatureSet{curFold};
        end
    end
end
