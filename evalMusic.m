% script for Jet's qualifying exam
clear all;

% ============= global setting ==================
N1 = 100;

doaDeg1 = rand(1,N1)*180-90;
err = zeros(1,N1);
ground = zeros(1,N1);
predict = zeros(1,N1);


for i = 1:N1
    
    ground(i) = doaDeg1(i);
    doaDeg = [ ground(i) 10 50.2];
    DoA       = doaDeg*pi/180; %DOA’s of signals in rad.
    P          = [1 1 1];
    K          = 5;
    d          = 0.5;  % attenna spacing
    N          = 1000;
    noise_var  = 1;
    RESOLUTION = 0.1;
    isCorrelated = 0;
    
    % ============= signal ==================
    
    X = ULASig( DoA,P ,K ,d , N , noise_var, isCorrelated  );
    % ============= MUSIC ==================
    
    r = length(DoA);
    hat = myMusic(X, r, RESOLUTION, d);
    predict(i) = hat(1);
    
    
    
    
    err(i) = ground(i) - predict(i);
    
end

errMean = mean(abs(err))
figure;
histogram(err);
figure;
scatter(ground, predict);