function x = Sig(sig)



theta = sig.theta;
Ac = sig.P;
fc = sig.f_c;
phi_c = sig.phi_c;
phi_i = sig.phi_c;

SNRdB = sig.SNRdB;
d = sig.d;
M = sig.M;
N = sig.N;
K = sig.K;

if strcmp(sig.geometry,'UCA')
    phi = sig.phi;  % for UCA
end
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
% sig.geometry = sig.geometry; %'UCA'
% sig.type = sig.type; %'digital'
% sig.mod = sig.mod; %'AM', 'PM'
% sig.intf = sig.intf; % 'chirp'

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
        if strcmp(sig.geometry,'ULA')
            A(m,k) = exp(1j*2*pi*D(m)*cos(theta(k)));
        elseif strcmp(sig.geometry,'UCA')            
            R = 40;    % radius (relateive, R/lamda) % DO NOT set it too small
            A(m,k) = exp(1j*2*pi*R*sin(theta(k))*cos(phi(k)-Gamma(m)));
        else
            disp('not implement yet');
        end
    end
end

% ------------- signal  -----------
S = zeros(K,N);
for k = 1:K
    %     if k == 1
    % first line
    for n = 1:N
        if strcmp(sig.type,'tone')
            S(k,n) = Ac(k) * exp(1j*2*pi*fc(k)*n + phi_c(k));
%             S(k,n) = Ac(k) * exp(1j*2*pi*fc(k)*n + rand*180/pi);
            % not possible for random
        elseif strcmp(sig.type,'digit')
            S(k,n) = sign(2*rand-1);
        end
    end
    % other are cycle shift of first
    %     else
    %         S(k,:) = circshift(S(1,:),floor(20*k));
    %         S(k,:) = circshift(S(1,:),floor(20*rand));  % introduce rand
    
    
    %     end
    
    % ------------- modulation  -----------
    if strcmp(sig.mod,'PS')
        temp = myPulseShape(S(k,:), 4, 4, 0.25,'sqrt');
        S(k,:) = temp(1:N);
    elseif strcmp(sig.mod,'AM')
        temp = myAM(S(k,:));
        S(k,:) = temp(1:N);
    elseif strcmp(sig.mod,'FM')
        temp = myFM(S(k,:));
        S(k,:) = temp(1:N);
    end
    
end


% ------------- noise & interference  -----------
Noise =  sqrt(2)/(10^(SNRdB/10))* (rand(M,N) + 1j*rand(M,N));

Intf = zeros(M,N);
for i = 1:M
    for j = 1:N
        Intf(i,j) = Ac(1) * exp(1j*2*pi*fc(1)*n + phi_i(1));
%         Intf(i,j) = Ac(1) * exp(1j*2*pi*fc(1)*n + rand*180/pi);
    end
end

x = A * S + Noise ;

end























