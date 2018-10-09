classdef testUnity < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        kFold;
        CostList;
        thresholdList;
        typeIerrorList;
        typeIIerrorList;
        numofRisk;
        numofNormal;
    end
    
    methods
        function obj = testUnity(costList,thresholdList, fold)
            obj.CostList = costList;
            obj.thresholdList = thresholdList;
            [obj.numofRisk, obj.numofNormal]=obj.ComputetheNumOfRiskandNormal(fold);
            obj.kFold =  size(fold.training, 2);
            for i = 1:length(thresholdList)*3
                if mod(i,3)== 1 % acc
                elseif mod(i,3) == 2 % typeI
                    obj.typeIerrorList = [obj.typeIerrorList i];
                elseif mod(i,3) == 0 % typeII
                    obj.typeIIerrorList = [obj.typeIIerrorList i];
                end
            end
        end
        function [numOfRisk, numOfNormal] = ComputetheNumOfRiskandNormal(obj, fold)
            Normal = 0;
            Risk = 0;
            Risk = sum(fold.training{1}(:,1));
            Risk = Risk + sum(fold.testData{1}(:,1));
            totalNum = size(fold.training{1},1) + size(fold.testData{1},1);
            Normal = totalNum - Risk;
            
            
            numOfRisk = Risk;
            numOfNormal = Normal;
        end
        function [newData] = reshapeData(obj,Data,totalFold)
            newData = [];
            thresholdListSize = size(obj.thresholdList, 2);
            for i = 1:thresholdListSize
                temp = [];
                for j = 1:totalFold
                    cutTime = i+(j-1)*thresholdListSize;
                    list(i,j) = cutTime;
                    temp(:,j) = Data(:,cutTime);
                end
                newData = [newData ;temp];
            end
        end
        function [output] = mergeResult(obj,modelObj)
            output = [];
            if ~iscell(modelObj)
                output = modelObj.getConfusionMatrixList();
            else
                for i = 1:size(modelObj, 2)
                    output = [output modelObj{i}.getConfusionMatrixList()];
                end
            end
            output = obj.reshapeData(output, size(modelObj, 2)*obj.kFold);
            
        end
        function [output] = mergeFS(obj,FeatureSet, modelObj)
            output = FeatureSet;
            
            if ~iscell(modelObj)
                localOutput = zero(size(modelObj.FeatureSet, 2), length(FeatureSet));
                for j = 1:size(modelObj.FeatureSet, 2)
                    selectFeature = modelObj.FeatureSet{j};
                    featureWeight = modelObj.FeatureWeightSet{j};
                    for k = 1:length(selectFeature)
                        index = find(FeatureSet== selectFeature(k));
                        localOutput(j, index) = featureWeight(k);
                    end
                end
                output = [output; localOutput];
            else
                for i = 1:size(modelObj, 2)
                    localOutput = zeros(size(modelObj{i}.FeatureSet, 2), length(FeatureSet));
                    
                    for j = 1:size(modelObj{i}.FeatureSet, 2)
                        selectFeature = modelObj{i}.FeatureSet{j};
                        featureWeight = modelObj{i}.FeatureWeightSet{j};
                        for k = 1:length(selectFeature)
                            index=find(FeatureSet == selectFeature(k));
                            localOutput(j, index) = featureWeight(k);
                        end
                    end
                    output = [output; localOutput];
                end
            end
        end
        function [Cost,typeI,typeII, average, chooseIndex ] = calculateCost(obj, Data)
            %             CostList,RiskNum,NormalNum,typeIerrorList,typeIIerrorList
            average = zeros(3, size(obj.CostList, 2));
            chooseIndex=[];
            for k = 1:size(obj.CostList,2)
                for i =1:size(Data, 1)/3
                    curpenalty = (i-1)*3+1;
                    for j = 1:size(Data, 2)
                        misclassificationCost(i,j) = Data(1+curpenalty,j)*obj.numofRisk*obj.CostList(k) + Data(2+curpenalty,j)*obj.numofNormal;
                    end
                end
                totalCost = sum(misclassificationCost, 2);
                averageCost = mean(misclassificationCost, 2);
                %ans = totalCost/50;
                [curCost,index] = min(totalCost);
                %          index
                chooseIndex= [chooseIndex; index];
                Cost(k,:) = misclassificationCost(index, :);
                typeI(k, :) =  Data(1+(index-1)*3+1, :);
                typeII(k, :) = Data(2+(index-1)*3+1, :);
                average(1, k) = mean(Data((index-1)*3+1,:));
                average(2, k) = mean(Data((index-1)*3+2,:));
                average(3, k) = mean(Data((index-1)*3+3,:));
                average(4,k) = mean(averageCost(index));
            end
        end
        function [result] = countRank(obj,Data)
            % Data = 要用來計算CostTable的資料
            % costList = Cost的List，看使用者要計算的Cost有哪些
            rankList = [];
            %% 計算每一列的平均值
            % j = classifier數量
            % i = cost數量
            for j = 1:size(Data,1)
                for i = 1:size(Data,2)
                    aver(j,i) = mean(Data(j,i,:));
                end
            end
            
            %% 找出平均值最低在哪個model，然後拿他來當base跟其他model對每個cost的資料做singrankTest
            for i = 1:size(obj.CostList,2)
                [~,minindex] = min(aver(:,i));
                for j = 1:size(Data,1)
                    temp1 = reshape(Data(minindex,i,1:end),1,size(Data,3));
                    temp2 = reshape(Data(j,i,1:end),1,size(Data,3));
                     temp(j) = signrank(temp1,temp2);
%                     [~, temp(j)] = ttest2(temp1,temp2)   
                end
                rankList = [rankList; temp];
            end
            %% 將最後結果回傳
            result = rankList;
        end
        function getDetCurve(obj, zoomMatrix ,varargin)
            typeIerrorList = []
            typeIIerrorList = []
            accuracyList = []
            % design to plot at most 4 line
            % can rewrite to plot more
                        for i = 1:length(obj.thresholdList)*3
                            if mod(i,3)== 1 % acc
                                accuracyList = [accuracyList, i];
                            elseif mod(i,3) == 2 % typeI
                                typeIerrorList = [typeIerrorList i];
                            elseif mod(i,3) == 0 % typeII
                                typeIIerrorList = [typeIIerrorList i];
                            end
                        end
            line([0 1]', [0 1]', 'Color','black', 'DisplayName', 'EERLine')
            hold on
            %% draw DET curve
            styleList = {'b*--' 'mo--' 'r^--' 'c<--' 'ks--' 'y+--' 'gh--'};
            for i = 1:length(varargin)
                curData = obj.mergeResult(varargin{i});
                average = mean(curData, 2);
                average = [average ;1 ;0]
                typeIerrorList = [302 typeIerrorList 301]
                typeIIerrorList = [301 typeIIerrorList 302]
                plot(average(typeIerrorList, 1), average(typeIIerrorList,1), styleList{i}, 'DisplayName', varargin{i}{1}.DISCRIPTION);
                hold on;
                %% calculation of EER value
                %False Rejection Rate (FRR) : Type I Error Rate
                %False Acceptance Rate (FAR) : Type II Error Rate
                %Equal Error Rate (EER) by FAR=FRR
                FRR = average(obj.typeIerrorList, 1);
                FAR = average(obj.typeIIerrorList,1);
                tmp1=find (FRR-FAR<=0);
                tmps=length(tmp1);
                
                %line1
                x1  = [FRR(tmps) FRR(tmps+1)];
                y1  = [FAR(tmps) FAR(tmps+1)];
                %EERline
                x2 = [0 1];
                y2 = [0 1];
                %fit linear polynomial
                p1 = polyfit(x1,y1,1);
                p2 = polyfit(x2,y2,1);
                %calculate intersection
                x_intersect = fzero(@(x) polyval(p1-p2,x),3);
                y_intersect = polyval(p1,x_intersect);
                EER = x_intersect;
                EER_str = num2str(x_intersect);
                plot(x_intersect,y_intersect,'o', 'MarkerSize',5,'LineWidth',2,'Color','black','MarkerFaceColor','k','DisplayName',EER_str)
                if i==1
                    hold on;
                end
            end
            legend('show');
            xlabel('TypeI');
            ylabel('TypeII');
            set(gca,'YTick',[0:0.05:1]);
            set(gca,'XTick',[0:0.05:1]);
            axis(zoomMatrix);
           % legend('M0', 'M1', 'M2', 'M3')
    end

        function [Cost,typeI,typeII, average, chooseIndex ] = getCostPackage(obj, modelobj)
             data=obj.mergeResult(modelobj);
             [Cost,typeI,typeII, average, chooseIndex ] = obj.calculateCost(data);
        end
        function [output, descripe] = getWilcoxonTest(obj, varargin)
            %nargin
            AllCost=[];
            
            for i = 1:nargin-1
                curData = obj.mergeResult(varargin{i});
                AllCost(i,:,:) = obj.calculateCost(curData);
                descripe{i} = varargin{i}{1}.DISCRIPTION;
            end
            output = obj.countRank(AllCost);
        end
    end
    
end

