function x = VectorSensorSig(theta,phi,gamma,ita, P, fc,...
    phi_c, SNRdB, N, dt,K, sigType)

a = zeros(6,K);
for k = 1:K
    Theta = [ cos(theta(k))*cos(phi(k)) -sin(phi(k)); ...
        cos(theta(k))*sin(phi(k)) cos(phi(k)); ...
        -sin(theta(k))            0;...
        -sin(phi(k))              -cos(theta(k))*cos(phi(k));...
        cos(phi(k))               -cos(theta(k))*sin(phi(k));...
        0                         sin(theta(k))...
        ];
    g = [sin(gamma(k))*exp(1j*ita(k))   cos(gamma(k)) ].';
    a(:,k) = Theta * g;
end

A1 = a;

if strcmp(sigType,'tone')
    s = zeros(K, N);
    for t = 1:N
        for k = 1:K
            s(k, t) = sqrt(P(k)) * exp( 1j*(2*pi*fc(k)*t*dt+phi_c(k) ) );
        end
    end
elseif strcmp(sigType,'pulseShaping')

    s = []; 
    for k = 1:K
        bits = 2*round(rand(1,N))-1;
        temp = myPulseShape(bits ,4, 4, 0.25,'sqrt'); 
        temp = temp(1:N);
        s = [s;temp];
    end
    
elseif  strcmp(sigType,'AM') 
        s = []; 
    for k = 1:K
        bits = 2*round(rand(1,N))-1;
        temp = myAM(bits); 
        temp = temp(1:N);
        s = [s;temp];
    end
    
elseif strcmp(sigType,'FM') 
        s = []; 
    for k = 1:K
        bits = 2*round(rand(1,N))-1;
        temp = myFM(bits);
        temp = temp(1:N);
        s = [s;temp];
    end
    
    
end
    

n = sqrt(2)/(10^(SNRdB/10))  * (rand(6, N) + 1j *  rand(6, N));

x = zeros(6,N);   % 12*N
for t = 1:N
    x(:,t) = A1 * s(:,t) + n(:,t);
end

end