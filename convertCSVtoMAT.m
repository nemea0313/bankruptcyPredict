%% 清空殘值
clear;
clc;

%% 將 *.csv檔 存成*.mat檔
% 將檔案讀入
% [A,B,C] = xlsread('TEJ.csv');
% A = 全部數字資料
% B = 全部文字資料
% C = 全部資料(包含文字、數字)

% [rawDatasetChinaCGI_NumberOnly,~,rawDatasetChinaCGI] = xlsread('D:\Google 雲端硬碟\中國公司治理\data\2005-2014中國CGI年報.csv');
% [rawDatasetChinaFR_NumberOnly,~,rawDatasetChinaFR] = xlsread('D:\Google 雲端硬碟\中國公司治理\data\2005-2015FR年報-刪除金融建築業.csv');
[USExpDataset,~,~] = xlsread('F:\Handover-20160630\Dinda data\20171114_USExpDataset.csv');

% 將想要的資料存成 *.mat
% FileName = strcat('E:\Google 雲端硬碟\FA\冠安論文\評委意見\中國公司治理\data\Normalized_2005-2014中國CGI年報.mat');
% FileName = strcat('D:\Google 雲端硬碟\中國公司治理\data\2005-2015FR年報-刪除金融建築業.mat');
FileName = strcat('F:\Handover-20160630\Dinda data\20171114_USExpDataset.mat');

save (FileName);
