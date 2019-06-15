function x = Sig(theta, Ac, fc, phi, SNRdB, d, M, N, K)
% theta = [58.6   26.7 26.7 30.0    48.6]./180*pi;  % DoA
% Ac     = [1 1 1 0 0];  % carrier amplitute
% K     = length(nonzeros(a));
% fc    = [ 0.75 0.55 0.55 0.85 0.45 ];
% phi   = [ 0 0 0 0 0];
% SNRdB = 30;
% M      = 8;
% d      = 0.45;
% N      = 200;   % snapshot


isBias = 1;
optGeo = 'ULA'; %'UCA'
optSrcSig = 'tone'; %'digital'
optMod = 'PS'; %'AM', 'PM'
optIntf = 'tone'; % 'chirp'

% ULA
D = [0:M-1]*d;  % relative spacing, d/lamda < 0.5
if isBias
    % anttenna spacing offset
    D(1) = D(1) + 0.03;
    D(2) = D(2) + 0.05;
end
% UCA
Gamma = [0:M-1]*2*pi/M;

% ------------- array manifold -----------
A = zeros(M,K);
for m = 1:M
    for k = 1:K
        if strcmp(optGeo,'ULA')
            A(m,k) = exp(1j*2*pi*D(m)*cos(theta(k)));
        elseif strcmp(optGeo,'UCA')
            f_a = 0.1; % angular frequency
            R = 1.0;    % radius
            A(m,k) = exp(1j*2*pi*f_a*R*sin(theta(k))*cos(phi(k)-Gamma(m)));
        else
            disp('not implement yet');
        end
    end
end

% ------------- signal and noise -----------
S = zeros(K,N);
for k = 1:K
    %     if k == 1
    % first line
    for n = 1:N
        if strcmp(optSrcSig,'tone')
            S(k,n) = Ac(k) * exp(1j*2*pi*fc(k)*n + phi(k));
%             S(k,n) = Ac(k) * exp(1j*2*pi*fc(k)*n + rand*180/pi);
            % not possible for random
        elseif strcmp(optSrcSig,'digit')
            S(k,n) = sign(2*rand-1);
        end
    end
    % other are cycle shift of first
    %     else
    %         S(k,:) = circshift(S(1,:),floor(20*k));
    %         S(k,:) = circshift(S(1,:),floor(20*rand));  % introduce rand
    
    
    %     end
end

Noise =  sqrt(2)/(10^(SNRdB/10))* (rand(M,N) + 1j*rand(M,N));

Intf = zeros(M,N);
for i = 1:M
    for j = 1:N
        Intf(i,j) = Ac(1) * exp(1j*2*pi*fc(1)*n + phi(1));
%         Intf(i,j) = Ac(1) * exp(1j*2*pi*fc(1)*n + rand*180/pi);
    end
end


x = A * S + Noise ;

end























