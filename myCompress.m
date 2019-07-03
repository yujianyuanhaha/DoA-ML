function S = myCompress(x)
% compress data

x2 = [real(x);imag(x)];
S = x2 * x2';
S = tril(S);
S = nonzeros(S);
S = reshape(S,[],1);
% S = normalize(S)';
S = (S - mean(S))/std(S);
S = S';

end