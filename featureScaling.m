clc;
close all;
clear all;
load('F:\Handover-20160630\Dinda data\20171114_USExpDataset.mat', 'USExpDataset');
FR = [5:46];

for j = 1:size(FR, 2)
    max_X = max(USExpDataset(:, j+4));
    min_X = min(USExpDataset(:, j+4));
    for i = 1: size(USExpDataset,1)
        normX(i, j) = (USExpDataset(i, j+4) - min_X) / (max_X - min_X);  
    end
%     stdX(:, j) = zscore(USExpDataset(:, j+4), 0);
end
USExpDataset(:, 5:46) = normX;

save('F:\Handover-20160630\Dinda data\Norm_20171114_USExpDataset.mat','USExpDataset');
