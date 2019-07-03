function  [net, tr] = myTrain(nn, Xtrain, Ytrain)

net = fitnet(nn.hiddenLayerSize);

if nn.isRad &&  ~nn.isBayes
    numLayer = length(nn.hiddenLayerSize);
    for i = 1:numLayer
        net.layers{i}.transferFcn = 'radbas';
    end
    [net, tr] = train(net, Xtrain, Ytrain);
elseif ~nn.isRad && nn.isBayes
    net.trainFcn = 'trainbr';
    [net, tr] = train(net, Xtrain, Ytrain);
elseif nn.isRad && nn.isBayes
    numLayer = length(nn.hiddenLayerSize);
    for i = 1:numLayer
        net.layers{i}.transferFcn = 'radbas';
    end
    net.trainFcn = 'trainbr';
    [net, tr] = train(net, Xtrain, Ytrain);
else
    [net, tr] = train(net, Xtrain, Ytrain,'useGPU','yes');
end

end