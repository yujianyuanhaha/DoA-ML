function y = myPulseShape(x, sps, span, beta, shape)
% compress data
p = myRC(beta,span,sps,shape);
% K * N, N over time
[k1,k2] = size(x);
y = zeros(k1,k2*sps+length(p)-1);
for i = 1:k1
   temp = upsample(x(i,:), sps);
   y(i,:) = conv(temp,p);
end

end