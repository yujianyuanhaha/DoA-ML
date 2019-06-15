function [hat,music_spectrum] = myMusic(X, K, RESOLUTION,d)

[M,N] = size(X);

R      = X * X'/K; %Spatial covariance matrix
[Q ,D] = eig(R); %Compute eigendecomposition of covariance matrix
[D, I] = sort(diag(D),1,'descend'); %Find K largest eigenvalues
Q      = Q(:,I); %Sort the eigenvectors to put signal eigenvectors first
Qs     = Q(:,1:K); %Get the signal eigenvectors
Qn     = Q(:,K+1:M); %Get the noise eigenvectors
% MUSIC algorithm
% Define angles at which MUSIC “spectrum” will be computed
angles = (-90:RESOLUTION:90);
%Compute steering vectors corresponding values in angles
a1     = exp(-1j*2*pi*d*(0:M-1)'*sin([angles(:).']*pi/180));
for k = 1:length(angles)
    %Compute MUSIC “spectrum”
    music_spectrum(k) = (a1(:,k)'*a1(:,k)) / (a1(:,k)'* Qn * Qn'*a1(:,k));
end
% figure;
% plot(angles,abs(music_spectrum),"LineWidth",2)
% grid on
% title('MUSIC Spectrum')
% xlabel('Angle in degrees')

% ============= MUSIC 2/2 return the maximum ==================
x = abs(music_spectrum);
a=x(1:end-2);
b=x(2:end-1);
c=x(3:end);
locations = find(b>a & b>c)+1;
turnPoint = angles(locations);
turnPointValue = x(locations);
[~,I] = maxk(turnPointValue,3);
hat = turnPoint(I);

end