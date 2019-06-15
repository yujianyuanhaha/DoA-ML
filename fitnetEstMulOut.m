% refined code

close all;
clear all;
% =================== global setting ============
note = sprintf('2 output');
% ------ signal setting --------
% optSig = 'VectorSensor'
optSig = 'VectorSensor'
isSend = 1 % disable when debug
isRadBas = 0
isBayes = 0

id = 'est4';

tag = strcat(id,optSig);
tagData        = strcat(tag,'Data.mat');
tagLabel       = strcat(tag,'Label.mat');
tagTargetLabel = strcat(tag,'TargetLabel.mat');
tagSampTarget  = strcat('samp',tag,'Target.mat');
tagSampInput   = strcat('samp',tag,'Input.mat');


theta  = [58.6   26.7 26.7 30.0    48.6]./180*pi;  % DoA
P      = [1 1 1 0 0];
K      = length(nonzeros(P));
fc     = [ 0.75 0.55 0.55 0.85 0.45 ];
phi_c  = [ 0 0 0 0 0];
SNRdB  = 10;
N      = 200;   % snapshot


if strcmp(optSig ,'ULA')
    M      = 8;
    d      = 0.45;
    inputLen = M*(2*M+1);  % ToDo related to M
elseif strcmp(optSig ,'VectorSensor')
    phi    = ( [360-57.4 360-69.1 13.3 106.3 360-170.8])./180*pi;
    gamma    = [0     1/2   3/4  1/4      1/4]*pi;
    ita      = [0     0     0    1/2      1/2]*pi;
    dt       = 0.10;
    deltaT   = 5 * dt;
    inputLen = 78;
end


% ------ data setting --------
setN     = [40 40 40 40];
Sizefull = prod(setN);
allData  = zeros(Sizefull,inputLen);
% sampSize = round(Sizefull*0.2/100)*100;
sampSize = round(Sizefull);
targetID = [1,2,3,4];
numOut   = length(targetID);
% ------- iterator of data -----
theta1Rnd = rand(1,180) *pi;
theta2Rnd = rand(1,180) *pi;
theta3Rnd = rand(1,180) *pi;
theta4Rnd = rand(1,180) *pi;




% ------ neural network setting --------
hiddenLayerSize = [25 25 25];



% =================== synthesis data ============
if isempty(dir(tagData)) & isempty(dir(tagData))
    tic;
    i = 0;
    for theta1Ind = 1:1:setN(1)
        for theta2Ind =  1:1:setN(2)
            for theta3Ind =  1:1:setN(3)
                for theta4Ind =  1:1:setN(4)
%               for theta3Ind =  1:1:setN(3)
            
            i = i + 1;
            %  -- rewrite --
            theta(1) = theta1Rnd(theta1Ind);
            theta(2) = theta1Rnd(theta2Ind);
            theta(3) = theta1Rnd(theta3Ind);
            theta(4) = theta1Rnd(theta4Ind);
%              theta(3) = theta1Rnd(theta3Ind);
            
            % ------- generate data -----
            
            
            if strcmp(optSig ,'ULA')
                x = ULASig(theta, P, fc, phi_c, SNRdB, d, M, N, K);
            elseif strcmp(optSig ,'VectorSensor')
                sigType = 'tone';
                x = VectorSensorSig(theta,phi,gamma,ita, P, fc,...
                    phi_c, SNRdB, N, dt,K,sigType);
            end
            % ------- compress data -----
            S = myCompress(x);
            
            allData(i,:) = S;
            Label(i).K      = K;
            Label(i).theta  = theta;
            Label(i).fc     = fc;
            Label(i).a      = P;
            Label(i).N      = N;
            Label(i).SNRdB  = SNRdB;
            
            if strcmp(optSig,'VectorSensor')
                Label(i).phi    = phi;
                Label(i).gamma  = gamma;
                Label(i).ita    = ita;
                Label(i).dt     = dt;
                Label(i).deltaT = deltaT;
            end
            
            
            if mod(i,1000) == 0
                processMsg = sprintf('data generating %.2f %%', i*100.0/Sizefull);
                disp(processMsg);
                toc;
            end
        end
        end
    end
    end
    toc;
    
else
    disp("data found, load data");
    load(tagData);
    load(tagLabel);
    load(tagTargetLabel);
    load(tagSampInput);
    load(tagSampTarget);
    
end
% -------extract Label -----
temp = reshape([Label.theta],[5,Sizefull]);
temp = temp';
TargetLabel = temp(:,targetID);   % 1st sig
% ------- sample data -----
Inputs = allData(1:sampSize,:)';
Targets = TargetLabel(1:sampSize,:)';

save(tagData,'allData');
save(tagLabel,'Label');
save(tagTargetLabel,'TargetLabel');
save(tagSampInput,'Inputs');
save(tagSampTarget,'Targets');



% =================== train ============
net = fitnet(hiddenLayerSize);
tic;
if isRadBas &  ~isBayes
    numLayer = length(hiddenLayerSize);
    for i = 1:numLayer
        net.layers{i}.transferFcn = 'radbas';
    end
    [net, tr] = train(net, Inputs, Targets);
elseif ~isRadBas & isBayes
    net.trainFcn = 'trainbr';
    [net, tr] = train(net, Inputs, Targets);
elseif isRadBas & isBayes
    numLayer = length(hiddenLayerSize);
    for i = 1:numLayer
        net.layers{i}.transferFcn = 'radbas';
    end
    net.trainFcn = 'trainbr';
    [net, tr] = train(net, Inputs, Targets);
else
    [net, tr] = train(net, Inputs, Targets,'useGPU','yes');
end
toc;
% =================== test ============
tInd = tr.testInd;
predict = net(Inputs(:, tInd));


ground = Targets(:,tInd);
% =================== eval ============
tstPerform = perform(net, ground, predict)
errors = abs(gsubtract(ground, predict));
if numOut > 1
    mse1    = immse(ground(1,:), predict(1,:))
    mse2    = immse(ground(2,:), predict(2,:))
     mse3    = immse(ground(3,:), predict(3,:))
    errM = mean(errors')
    eerMdeg = errM*180/pi
    errVar = var(errors')
else
    
    mse    = immse(ground, predict)
    errM = mean(errors)
    eerMdeg = errM*180/pi
    errVar = var(errors)
end




save(strcat(id,'errors.mat'),'errors');
save(strcat(id,'ground.mat'),'ground');
save(strcat(id,'predict.mat'),'predict');









% ======== save fig & send email ============
figure;
if numOut > 1
    histogram(errors(1,:));
    hold on;
    histogram(errors(2,:))
else
    histogram(errors);
end
clear title xlabel ylabel;
title('estimate error histogram');
xlabel('estimate error (rad)');
ylabel('count');
grid on;
saveas(gcf,'hist.png');



figure;
[hy1, hx1] = ecdf(errors(1,:));
plot(hx1,hy1);
hold on;
[hy2, hx2] = ecdf(errors(2,:));
plot(hx2,hy2);
hold on;
clear title xlabel ylabel;
title('CDF of error');
xlabel('estimate error (rad)');
ylabel('P(X < x)');
grid on;
saveas(gcf,'cdf.png');
save(strcat(id,'hx1.mat'),'hx1');
save(strcat(id,'hy1.mat'),'hy1');
save(strcat(id,'hx2.mat'),'hx2');
save(strcat(id,'hy2.mat'),'hy2');

figure;
if numOut > 1
    scatter(ground(1,:),predict(1,:));
    hold on;
    scatter(ground(2,:),predict(2,:),'*');
else
    scatter(ground,predict);
end
clear title xlabel ylabel;
grid on;
title('ground vs ML estimate')
xlabel('estimate ML');
ylabel('ground');
saveas(gcf,'scatter.png');

% neural network topoloy
jframe = view(net);
hFig = figure('Menubar','none', 'Position',[100 100 1200 166]);
jpanel = get(jframe,'ContentPane');
[~,h] = javacomponent(jpanel);
set(h, 'units','normalized', 'position',[0 0 1 1])
jframe.setVisible(false);
jframe.dispose();
set(hFig, 'PaperPositionMode', 'auto')
saveas(hFig, 'nn.png')
close(hFig)

figure;
plotperform(tr);
saveas(gcf,'perf.png');

figure;
plottrainstate(tr);
saveas(gcf,'trainState.png');

figure;
if numOut > 1
    ploterrhist(errors(1,:),'sig1',errors(2,:),'sig2');
else
    ploterrhist(errors);
end
saveas(gcf,'hist2.png');

figure;
if numOut > 1
    plotregression(ground(1,:), predict(1,:),'sig1',...
    ground(2,:), predict(2,:),'sig2');
else
    plotregression(ground, predict);
end

saveas(gcf,'regression.png');

title = strcat('(Dell) ',optSig, ' ',note, sprintf(' mean error %.4f degree',eerMdeg));
setting = sprintf('SNRdB=%.2f; P1=%.2f, P2=%.2f, P3=%.2f, P4=%.2f, P5=%.2f, ',...
    SNRdB,P(1),P(2),P(3),P(4),P(5));
nnSetting = sprintf('TrainSize = %d, time = %.2f, trainFcn = %s, isRadBas = %d, isBayes = %d ', ...
    sampSize, tr.time(end),tr.trainFcn, isRadBas,isBayes);
content = strcat( setting,newline,nnSetting);
attachment = {'hist.png','scatter.png','nn.png',...
    'perf.png','trainState.png','hist2.png',...
    'cdf.png',strcat(id,'errors.mat'),strcat(id,'ground.mat'),...
    strcat(id,'predict.mat'),strcat(id,'hx1.mat'),strcat(id,'hy1.mat'),...
    strcat(id,'hx2.mat'),strcat(id,'hy2.mat'),'regression.png',mfilename};
if isSend
    sendEmail(title,content,attachment);
end