classdef (Abstract) modelToolkit < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties  (Abstract)
        KFOLD;
        model;
        ConfusionMatrix;
        DISCRIPTION;
        % penalty can not be used with prob;
        penaltyList;
        prob;
        thresholdList;
    end
    
    methods
        output=trainModel(obj)
        output=testModel(obj)
        output=getConfusionMatrix(obj)
    end
    
end

