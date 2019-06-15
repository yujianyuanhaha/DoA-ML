% script for Jet's qualifying exam
clear all;

% ============= global setting ==================
N1 = 200;
optMethod = 'ESPRIT';

N          = 4000;  % snapshot
d          = 0.2;  % attenna spacing
M          = 10;
noise_var  = 1;
RESOLUTION = 0.1;
isCorrelated = 0;

doaDeg1 = rand(1,N1)*180-90;
err = zeros(1,N1);
ground = zeros(1,N1);
predict = zeros(1,N1);

tic;

for i = 1:N1
    
    ground(i) = doaDeg1(i);
    doaDeg = [ ground(i) 10 50.2];
    DoA       = doaDeg*pi/180; %DOA’s of signals in rad.
    P          = [1 1 1];
  
    % ============= signal ==================
    
    X = ULASig( DoA,P ,N ,d , M , noise_var, isCorrelated  );
    K = length(DoA);
    % ============= MUSIC ==================
    if strcmp(optMethod,'MUSIC') 
        
    [hat,music_spectrum] = myMusic(X, K, RESOLUTION, d);
    elseif strcmp(optMethod,'ESPRIT')
        Delta = M/2;
        [hat,music_spectrum] = myESPRIT(X, K, d, Delta);
    end

    predict(i) = hat(1);
    err(i) = ground(i) - predict(i);

    if mod(i,M/10)
        disp( sprintf("prccess %.2f %%",i*100.0/M) );    
    toc;

    end
    
end



toc;
errMean = mean(abs(err))


if isempty(dir('./Data'))
    mkdir Data
end
if isempty(dir('./Fig'))
    mkdir Fig
end


figure;
histogram(err);
figure;
scatter(ground, predict);


save('./Data/musicScatterGround.mat','ground');
save('./Data/musicScatterPredict.mat','predict')