% refined code

close all;
clear all;
% =================== global setting ============
note = sprintf('QE ');
% ------ signal setting --------
optSig   = 'ULA'
% optSig   = 'ULA'
isSend   = 1 % disable when debug
isRadBas = 0
isBayes  = 0
% isAM     = 0
% isFM     = 0
% isPS     = 0
isRand   = 1

sigType = 'tone'
% pulseShaping, AM, FM

id = 'test';
tag = strcat(id,optSig);
tagData        = strcat(tag,'Data.mat');
tagLabel       = strcat(tag,'Label.mat');
tagTargetLabel = strcat(tag,'TargetLabel.mat');
tagSampTarget  = strcat('samp',tag,'Target.mat');
tagSampInput   = strcat('samp',tag,'Input.mat');


theta  = [58.6   26.7 26.7 30.0    48.6]./180*pi;  % DoA
P      = [1 1 1 1 1];
K      = length(nonzeros(P));
fc     = [ 0.55 0.55 0.55 0.85 0.45 ];
phi_c  = [ 0 0 0 0 0];
SNRdB  = 10;
N      = 400;   % snapshot


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
setN     = [40 40];
Sizefull = prod(setN);
allData  = zeros(Sizefull,inputLen);
sampSize = round(Sizefull*0.8/100)*100;
targetID = 1;

% ------- iterator of data -----

if isRand
    theta1Rnd = rand(1,180) *pi;
    theta2Rnd = rand(1,180) *pi;
%     phi1Rnd = rand(1,180)*pi;
%     phi2Rnd = rand(1,180) *pi;
else
    theta1Rnd = [0:1:179]/180*pi;
    theta2Rnd = [0:1:179]/180*pi;
end

% ------ neural network setting --------
hiddenLayerSize = [25 25 25];



% =================== synthesis data ============
if isempty(dir(tagData)) & isempty(dir(tagData))
    tic;
    i = 0;
    for theta1Ind = 1:1:setN(1)
        for theta2Ind =  1:1:setN(2)
            
            i = i + 1;
            %  -- rewrite --
            theta(1) = theta1Rnd(theta1Ind);
            theta(2) = theta1Rnd(theta2Ind);
%             phi(1) = phi1Rnd(theta1Ind);
%             phi(2) = phi2Rnd(theta2Ind);
%                 theta(1) = theta1Ind;
%                 theta(2) = theta2Ind;
            
            % ------- generate data -----
            
            
            if strcmp(optSig ,'ULA')
                x = ULASig2(theta, P, fc, phi_c, SNRdB, d, M, N, K);
            elseif strcmp(optSig ,'VectorSensor')
                x = VectorSensorSig(theta,phi,gamma,ita, P, fc,...
                    phi_c, SNRdB, N, dt,K,sigType);
            end
            
           
            % ------- modulate AM/ Pulse Shaping -----
%             if isAM 
%                 x = myAM(x);
%             end
%             if isFM 
%                 x = myFM(x);
%             end
%             if isPS 
%                 x = myPulseShape(x ,4, 4, 0.25,'sqrt');           
%             end
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
    toc;
    
else
    disp('data found, load data');
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

if isRadBas &&  ~isBayes
    numLayer = length(hiddenLayerSize);
    for i = 1:numLayer
        net.layers{i}.transferFcn = 'radbas';
    end
    [net, tr] = train(net, Inputs, Targets);
elseif ~isRadBas && isBayes
    net.trainFcn = 'trainbr';
    [net, tr] = train(net, Inputs, Targets);
elseif isRadBas && isBayes
    numLayer = length(hiddenLayerSize);
    for i = 1:numLayer
        net.layers{i}.transferFcn = 'radbas';
    end
    net.trainFcn = 'trainbr';
    [net, tr] = train(net, Inputs, Targets);
else
    [net, tr] = train(net, Inputs, Targets,'useGPU','yes');
end

% =================== test ============
tInd = tr.testInd;
tstOutputs = net(Inputs(:, tInd));

ground = Targets(tInd);

% =================== eval ============
tstPerform = perform(net, ground, tstOutputs)
errors = abs(gsubtract(ground, tstOutputs));
mse    = immse(ground, tstOutputs)
errM = mean(errors)
eerMdeg = errM*180/pi
errVar = var(errors)

save(strcat(id,'errors.mat'),'errors');
save(strcat(id,'ground.mat'),'ground');
save(strcat(id,'tstOutputs.mat'),'tstOutputs');

% ======== save fig & send email =====s=======
figure;
histogram(errors);
clear title xlabel ylabel;
title('estimate error histogram');
xlabel('estimate error (rad)');
ylabel('count');
grid on;
saveas(gcf,'hist.png');


% cdf of error
figure;
[hy, hx] = ecdf(errors);
plot(hx,hy);
clear title xlabel ylabel;
title('CDF of error');
xlabel('estimate error (rad)');
ylabel('P(X < x)');
grid on;
saveas(gcf,'cdf.png');
save(strcat(id,'hx.mat'),'hx');
save(strcat(id,'hy.mat'),'hy');

figure;
scatter(ground,tstOutputs);
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
ploterrhist(errors);
saveas(gcf,'hist2.png');

figure;
plotregression(ground, tstOutputs);
saveas(gcf,'regression.png');

title = strcat('(Dell) ',optSig, ' ',note, sprintf(' mean error %.4f deg, mse %.4f deg',eerMdeg,mse));
setting = sprintf('SNRdB=%.2f; P1=%.2f, P2=%.2f, P3=%.2f, P4=%.2f, P5=%.2f, ',...
    SNRdB,P(1),P(2),P(3),P(4),P(5));
nnSetting = sprintf('TrainSize = %d, time = %.2f, trainFcn = %s, isRadBas = %d, isBayes = %d ,%s', ...
    sampSize, tr.time(end),tr.trainFcn,isRadBas, isBayes,sigType);
% content = strcat( setting,newline,nnSetting);
 content = strcat( setting,nnSetting);
attachment = {'hist.png','cdf.png','scatter.png','nn.png',...
    'perf.png','trainState.png','hist2.png',...
    'regression.png',strcat(id,'errors.mat'),strcat(id,'ground.mat'),strcat(id,'tstOutputs.mat'),...
    strcat(id,'hx.mat'),strcat(id,'hy.mat'),mfilename};
if isSend
    sendEmail(title,content,attachment);
end