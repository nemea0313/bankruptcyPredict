%% �M�Ŵݭ�
clear;
clc;

%% �N *.csv�� �s��*.mat��
% �N�ɮ�Ū�J
% [A,B,C] = xlsread('TEJ.csv');
% A = �����Ʀr���
% B = ������r���
% C = �������(�]�t��r�B�Ʀr)

% [rawDatasetChinaCGI_NumberOnly,~,rawDatasetChinaCGI] = xlsread('D:\Google ���ݵw��\���ꤽ�q�v�z\data\2005-2014����CGI�~��.csv');
% [rawDatasetChinaFR_NumberOnly,~,rawDatasetChinaFR] = xlsread('D:\Google ���ݵw��\���ꤽ�q�v�z\data\2005-2015FR�~��-�R�����īؿv�~.csv');
[USExpDataset,~,~] = xlsread('F:\Handover-20160630\Dinda data\20171114_USExpDataset.csv');

% �N�Q�n����Ʀs�� *.mat
% FileName = strcat('E:\Google ���ݵw��\FA\�a�w�פ�\���e�N��\���ꤽ�q�v�z\data\Normalized_2005-2014����CGI�~��.mat');
% FileName = strcat('D:\Google ���ݵw��\���ꤽ�q�v�z\data\2005-2015FR�~��-�R�����īؿv�~.mat');
FileName = strcat('F:\Handover-20160630\Dinda data\20171114_USExpDataset.mat');

save (FileName);
