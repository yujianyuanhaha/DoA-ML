% script for Jet's qualifying exam
clear all;

% ============= global setting ==================
doaDeg = [20 25 33]
DoA       = doaDeg*pi/180; %DOA’s of signals in rad.
P          = [1 1 1];
M          = 10;
d          = 0.2;  % attenna spacing
N          = 2000;  
noise_var  = 0.01;
RESOLUTION = 0.1;
isCorrelated = 0;
K = length(DoA);


% ============= signal ==================

X = ULASig( DoA,P ,N ,d , M , noise_var, isCorrelated  );
% ============= MUSIC ==================

% [DoA_hat,music_spectrum] = myMusic(X, K, RESOLUTION, d);
% DoA_hat
% music_spectrum = abs(music_spectrum);
% figure;
% plot(music_spectrum);


% ============= ESPRIT ==================
Delta = 4;
[DoA_hat] = myESPRIT(X, K, d, Delta);
DoA_hat



