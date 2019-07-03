function y = myAM(x)
% compress data

% K * N, N over time
[k1,k2] = size(x);
y = zeros(k1,k2);
Ac = [2.0, 3.0,4.0,5.0,6.0,7.0]; % amplitude
fc = 0.10*ones(1,k1);
phi_c = 0.00*ones(1,k1);

for i = 1:k1
    for j = 1:k2
        y(i,j) = (1+x(i,j)/Ac(i)) * cos(2*pi*fc(i)*j+phi_c(i));
    end
end

end