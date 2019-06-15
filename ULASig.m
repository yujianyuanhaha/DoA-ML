function X = ULASig(doas,P ,N ,d , M , noise_var, isCorrelated  )
% K - num of source
% M - num of attena
% K - num of snapshot


K = length(doas);

% Steering vector matrix. Columns will contain the steering vectors
% of the K signals
A = exp(-1j*2*pi*d*(0:M-1)'*sin([doas(:).']));
% Signal and noise generation
if isCorrelated
    % Generate random BPSK symbols for each of ther signals
    sig = round(rand(1,N))*2-1;
%     sig = [sig;sig;sig];
    sig = repmat(sig,K,1);
else
    sig = round(rand(K,N))*2-1;
end

noise = sqrt(noise_var/2) * (randn(M,N)+1j*randn(M,N)); %Uncorrelated noise
X = A * diag(sqrt(P)) * sig + noise; %Generate data matrix

end