% script for Jet's qualifying exam
clear all;

% ============= global setting ==================
doas       = [-30 5.1 50.2]*pi/180; %DOA’s of signals in rad.
P          = [1 1 1];
K          = 5;
d          = 0.5;  % attenna spacing
N          = 100;
noise_var  = 1;
RESOLUTION = 0.1;
isCorrelated = 1;



% ============= signals ==================
r = length(doas);

% Steering vector matrix. Columns will contain the steering vectors
% of the r signals
A = exp(-1j*2*pi*d*(0:N-1)'*sin([doas(:).']));
% Signal and noise generation
if isCorrelated
    % Generate random BPSK symbols for each of ther signals
    sig = round(rand(1,K))*2-1;
    sig = [sig;sig;sig];
   
else
    sig = round(rand(r,K))*2-1;
end



noise = sqrt(noise_var/2) * (randn(N,K)+1j*randn(N,K)); %Uncorrelated noise
X = A * diag(sqrt(P)) * sig + noise; %Generate data matrix


% ============= MUSIC 1/2 generate 'spectrun' ==================

R      = X * X'/K; %Spatial covariance matrix
[Q ,D] = eig(R); %Compute eigendecomposition of covariance matrix
[D, I] = sort(diag(D),1,'descend'); %Find r largest eigenvalues
Q      = Q(:,I); %Sort the eigenvectors to put signal eigenvectors first
Qs     = Q(:,1:r); %Get the signal eigenvectors
Qn     = Q(:,r+1:N); %Get the noise eigenvectors
% MUSIC algorithm
% Define angles at which MUSIC “spectrum” will be computed
angles = (-90:RESOLUTION:90);
%Compute steering vectors corresponding values in angles
a1     = exp(-1j*2*pi*d*(0:N-1)'*sin([angles(:).']*pi/180));
for k = 1:length(angles)
    %Compute MUSIC “spectrum”
    music_spectrum(k) = (a1(:,k)'*a1(:,k)) / (a1(:,k)'* Qn * Qn'*a1(:,k));
end
figure;
plot(angles,abs(music_spectrum),"LineWidth",2)
grid on
title('MUSIC Spectrum')
xlabel('Angle in degrees')

% ============= MUSIC 2/2 return the maximum ==================
% [~,I] = maxk(music_spectrum,3)
x = abs(music_spectrum);
a=x(1:end-2);
b=x(2:end-1);
c=x(3:end);
locations = find(b>a & b>c)+1;
turnPoint = angles(locations);
turnPointValue = x(locations);
[~,I] = maxk(turnPointValue,3);
hat = turnPoint(I);
