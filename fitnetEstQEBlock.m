% refined code

close all;
clear all;

if isempty(dir('./Data'))
    mkdir Data
end
if isempty(dir('./Fig'))
    mkdir Fig
end

% =================== global setting ============

nn.isRad = 0;
nn.isBayes  = 0;
nn.hiddenLayerSize = [25 25 25];

Case.note   = sprintf('snapshot & SNR');
Case.isSend = 0 % disable when debug

sig.caseID   = 'test';
sig.geometry = 'ULA'
sig.isRand   = 1;
sig.type     = 'tone'
sig.mod      = 'none'
sig.intf     = 'none'  % single tone

sig.theta  = [58.6   26.7 26.7 36.7    48.6]./180*pi;  % DoA
sig.P      = [1 1 1 1 1];
sig.K      = length(nonzeros(sig.P));   % num of source
sig.f_c    = [ 0.55 0.65 0.75 0.85 0.45 ];
% sig.phi_c  = [ 0 0 0 0 0];
sig.phi_c  = [58.6   26.7 26.7 30.0    48.6]./180*pi;  % DoA
sig.SNRdB  = 5;
sig.N      = 500;   % snapshot
sig.M      = 6;  % num of Anntenna
sig.setN   = [90 90];

target.ID   = [1,2];
target.Para = 'theta';

tag            = sig.caseID;
tagData        = strcat(tag,'Data.mat');

if isempty(dir(tagData)) & isempty(dir(tagData))    
    [allData, rawData, Label, sig] = generateData(sig,target);
else    
    disp('data found, load data ...');
    load(strcat('./Data/',tagData));
    load(strcat('./Data/',sig.caseID,'RawData.mat'));
    load(strcat('./Data/',sig.caseID,'Label.mat'));
end

% ============== extract Label and Split Data====================

nn.trainRatio = 0.80;
Sizefull      = length(allData(:,1));
trainSize     = round(Sizefull*nn.trainRatio/100)*100;

% -------  extract Label -----
if strcmp(target.Para,'theta')
    temp = reshape([Label.theta],[sig.K,Sizefull]);
elseif strcmp(target.Para,'phi')
    temp = reshape([Label.phi],[sig.K,Sizefull]);
end
temp = temp';
Y = temp(:,target.ID);   % 1st sig
% ------- split data -----
Xtrain = allData(1:trainSize,:)';
Ytrain = Y(1:trainSize,:)';

Xtest = allData(trainSize:Sizefull,:)';
Ytest = Y(trainSize:Sizefull,:)';

save(strcat('./Data/',sig.caseID,'Y.mat'),      'Y'    );
save(strcat('./Data/',sig.caseID,'Xtrain.mat'),'Xtrain');
save(strcat('./Data/',sig.caseID,'Ytrain.mat'),'Ytrain');
save(strcat('./Data/',sig.caseID,'Xtest.mat'),'Xtest');
save(strcat('./Data/',sig.caseID,'Ytest.mat'),'Ytest');


% ======================= train ================================
[net, tr] = myTrain(nn, Xtrain, Ytrain);

% =================== test ============
% tInd = tr.testInd;
% predict = net(Xtrain(:, tInd));
% ground = Ytrain(tInd);


predict = net(Xtest);
ground = Ytest;

% =================== eval ============
predictPerformance = perform(net, ground, predict);
errors             = abs(gsubtract(ground, predict));
mse                = immse(ground, predict);
errM               = mean(errors);
eerMdeg            = errM*180/pi
RMSE               = sqrt(mean((errors).^2));
RMSE               = RMSE *180/pi
errVar             = var(errors);

save(strcat('./Data/',sig.caseID,'errors.mat'),'errors');
save(strcat('./Data/',sig.caseID,'ground.mat'),'ground');
save(strcat('./Data/',sig.caseID,'predict.mat'),'predict');

% ======== save fig & send email =====s=======
myPlot(eerMdeg,RMSE,errors,ground,predict, sig, nn, Case, net, tr);