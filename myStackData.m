function out = myStackData(A, stackNum)



% e.g 1000*16*924 -> 1000*(16*stackNum)*924
% A should be 2D
[N,M] = size(A);
space = (stackNum-1)/2;  % assume stackNum as odd
out = zeros([N,M*stackNum]);

if space < 1
    out = A;
end

for i = 1:N
    
    i1 = mod(i-space+N,N);
    if i1 == 0
        i1 = N;
    end
    temp2 = A(i1,:);
    for j = i-space+N+1 : i+space+N
        i2 = mod(j,N);
        if  i2 == 0
            i2 = N;
        end
        temp2 = [temp2, A(i2,:) ];
    end
    %         temp2 = (temp2 - np.mean(temp2))/(np.max(temp2)-np.min(temp2))
    out(i,:) = temp2;
end


end