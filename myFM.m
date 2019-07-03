function y = myFM(x)
% compress data

% K * N, N over time
[k1,k2] = size(x);

fs = [0.20, 0.30, 0.40, ...
      0.50, 0.60, 0.70]; % amplitude
fc = 0.10*ones(1,k1);
freqdev = 0.10*ones(1,k1);

temp = fmmod(real(x(1,:)),fc(1),fs(1),freqdev(1))+1j*fmmod(imag(x(1,:)),fc(1),fs(1),freqdev(1));
y = zeros(k1,length(temp));

for i = 1:k1
    
        % single tone could apply FM
        y(i,:) = fmmod(real(x(i,:)),fc(i),fs(i),freqdev(i))+1j*fmmod(imag(x(i,:)),fc(i),fs(i),freqdev(i));
    
end

end