% script for Jet's qualifying exam
clear all;
close all;

% ============= global setting ==================
N1 = 10;
optMethod = 'ESPRIT'

N          = 4000;  % snapshot
d          = 0.2;  % attenna spacing
M          = 10;
noise_var  = 0.01;
RESOLUTION = 0.1;
isCorrelated = 0;

doaDeg1 = rand(1,N1)*90;


tic;

err = [];
ground = [];
predict = [];
for i = 1:N1
    
   
    doaDeg = [ doaDeg1(i) 10 20];
    doaDeg = sort(doaDeg);
%     doaDeg
    ground = [ground;doaDeg];

    
    DoA       = doaDeg*pi/180; %DOA’s of signals in rad.
    P          = [1 1 1];
    
    K = length(DoA);
    

    
    
  
    % ============= signal ==================
    
    X = ULASig( DoA,P ,N ,d , M , noise_var, isCorrelated  );
   
    % ============= MUSIC ==================
    if strcmp(optMethod,'MUSIC')         
        [hat,music_spectrum] = myMusic(X, K, RESOLUTION, d);
        % some time not full rank !
        
        hat = sort(hat);
        predict = [predict;hat];
        
    elseif strcmp(optMethod,'ESPRIT')
        Delta = 4;
        [hat] = myESPRIT(X, K, d, Delta);
        hat = sort(hat)
        predict = [predict;hat];

    end

   
    err = ground - predict;

    if mod(i,M/10)
        disp( sprintf('prccess %.2f %%',i*100.0/M) );    
    toc;

    end
    
end


toc;
errMean = mean(abs(err));


if isempty(dir('./Data'))
    mkdir Data
end
if isempty(dir('./Fig'))
    mkdir Fig
end


err = err';
ground = ground';
predict = predict';

figure;
histogram(err(1,:));
figure;
scatter(ground(1,:), predict(1,:));



% save('./Data/musicScatterGround.mat','ground');
% save('./Data/musicScatterPredict.mat','predict')