% Ref Wong, Kainam T., and Michael D. Zoltowski.
% "Uni-vector-sensor ESPRIT for multisource azimuth, elevation, and polarization estimation."
% IEEE Transactions on Antennas and Propagation 45.10 (1997): 1467-1474.

% "resolve up to five completely polarized uncorrelated monochromatic sources
% from near field or far field"


% uncorrelated example from paper
% theta: elevation angle, [0,pi)   -> [-pi/2 +pi/2]
% theta = [58.6' 26.7' 51.4' 30.0' 48.6']
% phi: azimuth angle, [0,2*pi)  -> [0, 2*pi]
% phi   = [-57.4' -69.1' 13.3' 106.3' -170.8']


%========= global parameter ==========================
theta = [58.6   26.7 51.4 30.0    48.6]./180*pi;
%phi   = ( [-57.4 -69.1 13.3 106.3 -170.8])./180*pi;
phi   = ( [360-57.4 360-69.1 13.3 106.3 360-170.8])./180*pi;

gamma = [0     1/2   3/4  1/4      1/4]*pi;
ita   = [0     0     0    1/2      1/2]*pi;
K     = length(theta);

f = [ 0.75 0.55 0.65 0.85 0.45 ];

P     = [1 0 0 0 0];
Phase = rand(1,K)*2*pi;

dt     = 0.10;  % sample rate
N      = 200;    % number of snapshot
deltaT = 5 * dt;


% ======== calculate vector-sensor manifold: a ==========================
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


% ========= data model: calculate A1, A2, s(t_n), n(t_n), Phi ==========
A1 = a;
Phi = diag(exp(f*2*pi*1j*deltaT));
A2 = A1 * Phi;

s = zeros(K, N);
for t = 1:N
    for k = 1:K
        s(k, t) = sqrt(P(k)) * exp( 1j*(2*pi*f(k)*t*dt+Phase(k) ) );
    end
end

std = 0.0001;
n = std * rand(12, N) + 1j * std * rand(12, N);
% for k = 1:K
%     n2 = n1 * delayFactor(k);
% end
% n = [n1; n2];




Z = zeros(12,N);   % 12*N
for t = 1:N
    Z(:,t) = [A1;A2] * s(:,t) + n(:,t);
end

Z1 = Z(1:6,:);
Z2 = Z(7:12,:);



% ========= adopting ESPRIT to vector: calculate E1, E2 ==========
[V1, D1] = eig(Z1*Z1');
[d,ind1] = sort(diag(D1));
Vs1 = V1(:,ind1);
E1 = Vs1(:,7-K:6);   % 6*K

[V2, D2] = eig(Z2*Z2');
[d,ind2] = sort(diag(D2));
Vs2 = V2(:,ind2);
E2 = Vs2(:,7-K:6);   % 6*K

%T = pinv(A1)*E1;
% T2 = pinv(A2)*E2;  % looks not unique
% T = 1/2 * (pinv(A1)*E1 + pinv(A2)*E2);

%Psi = inv(E1'*E1)*(E1'*E2);
Psi = pinv(E1)*E2;

[Temp,Temp2] = eig(Psi);
%[w1, w2, W] = eig(Psi);
% T = W;
T = inv(Temp);

% =========Estimation: calculate A1_hat theta_hat phi_hat ==========

% A1_hat = E1 * inv(T);
A1_hat = 1/2 * ( E1 * (inv(T)\eye(K)) + E2 * (inv(T)\eye(K)) * inv(Phi));

p = zeros(3,K);
theta_hat = zeros(1,K);
phi_hat = zeros(1,K);

for k = 1:K
    a_hat  = A1_hat(:,k);
    e_hat  = a_hat(1:3,:);
    h_hat  = a_hat(4:6,:);
    p(:,k) = real( cross(e_hat/norm(e_hat), conj(h_hat)/norm(h_hat) ) );
    p(:,k) = p(:,k)/norm(p(:,k));  % notice
    
    u_hat = p(1,k);
    v_hat = p(2,k);
    w_hat = p(3,k);
    
   % theta_hat(k) = acos((w_hat)) - pi/2;
   
   % try cart to sph
   [theta_hat(k), phi_hat(k)] = cart2sph(u_hat,v_hat, w_hat );
   
   % [-pi/2 +pi/2]
    % matlab 'acos()' returns values in the interval [0,pi]
    % phi_hat(k)   = atan((v_hat)/(u_hat)) + pi*3/2; 
    % [0, 2*pi]
    % matlab 'atan()' returns values in the interval [-pi/2,pi/2]
    
    % TODO
end


theta_hat_abs = abs(theta_hat);

phi_hat_abs = abs(phi_hat);


% refine to be brief


