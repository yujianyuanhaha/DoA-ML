function x = ULASig(theta, a, fc, phi, SNRdB, d, M, N, K)
% theta = [58.6   26.7 26.7 30.0    48.6]./180*pi;  % DoA
% a     = [1 1 1 0 0];
% K     = length(nonzeros(a));
% fc    = [ 0.75 0.55 0.55 0.85 0.45 ];
% phi   = [ 0 0 0 0 0];
% SNRdB = 30;
% M      = 8;
% d      = 0.45;
% N      = 200;   % snapshot

x = zeros(M,N);
bias = 1;
D = [0:M-1]*d;

for m = 1:M
    for n = 1:N
        temp = 0;
        for k = 1:K
  
            if bias                
                D(1) = D(1) + 0.03;
                D(2) = D(2) + 0.05;
            end
            
            temp  =  temp + a(k) * exp(1j*2*pi*fc(k)*n + phi(k)) ...
                * exp( 1j*2*pi*D(m)*cos(theta(k))*1) ; 
        end
        noise =  sqrt(2)/(10^(SNRdB/10))* (rand + 1j*rand);
        x(m,n) = temp + + noise;
    end
end

end























