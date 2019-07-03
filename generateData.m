function [allData, rawData, Label, sig] = generateData(sig,target)

% =================== Sig Block ============
% IN: Case, sig
% Out: X, Y

if strcmp(sig.geometry ,'ULA')
    sig.d         = 0.40;
    sig.newLength = sig.M*(2*sig.M+1);  % ToDo related to sig.M
elseif strcmp(sig.geometry ,'UCA')
    sig.d         = 0.40;
    sig.newLength = sig.M*(2*sig.M+1);  % ToDo related to sig.M
    sig.phi = ( [360-57.4 360-69.1 13.3 106.3 360-170.8])./180*pi;  % ToDo related to sig.M  
elseif strcmp(sig.geometry ,'VectorSensor')
    sig.phi      = ( [360-57.4 360-69.1 13.3 106.3 360-170.8])./180*pi;
    sig.gamma    = [0     1/2   3/4  1/4      1/4]*pi;
    sig.ita      = [0     0     0    1/2      1/2]*pi;
    sig.dt       = 0.10;
    sig.deltaT   = 5 * sig.dt;
    sig.newLength = 78;  % 12*13/2
end

% ------ data setting --------

Sizefull = prod(sig.setN);
allData  = zeros(Sizefull,sig.newLength);
rawData  = zeros(Sizefull,sig.M,sig.N);



% ------- iterator of data -----

if sig.isRand
    if strcmp(target.Para,'theta')
        theta1Rnd = rand(1,180) *pi;
        theta2Rnd = rand(1,180) *pi;
    elseif strcmp(target.Para,'phi')
        phi1Rnd = rand(1,180)*pi;
        phi2Rnd = rand(1,180) *pi;
    end
    %     theta2Rnd = theta1Rnd+ 1*pi/180; % overlap, so as mess num of source
    
else
    theta1Rnd = [0:1:179]/180*pi;
    theta2Rnd = [0:1:179]/180*pi;
end



% =================== synthesis data ============

tic;
i = 0;
for theta1Ind = 1:1:sig.setN(1)
    for theta2Ind =  1:1:sig.setN(2)
        
        i = i + 1;
        %  -- rewrite --
        if strcmp(target.Para,'theta')
            sig.theta(1) = theta1Rnd(theta1Ind);
            sig.theta(2) = theta1Rnd(theta2Ind);
        elseif strcmp(target.Para,'phi')
            sig.phi(1) = phi1Rnd(theta1Ind);
            sig.phi(2) = phi2Rnd(theta2Ind);
        end
        %                 sig.theta(1) = theta1Ind;
        %                 sig.theta(2) = theta2Ind;
        
        % ------- generate data -----
        
        
        if strcmp(sig.geometry ,'ULA') ||strcmp(sig.geometry ,'UCA')
            x = Sig(sig);
        elseif strcmp(sig.geometry ,'VectorSensor')
            x = VectorSensorSig(sig.theta,sig.phi,sig.gamma,sig.ita, sig.P, sig.f_c,...
                sig.phi_c, sig.SNRdB, sig.N, sig.dt,sig.K,sig.type);
        end

        % ------- compress data -----
        S = myCompress(x);
        
        allData(i,:)    = S;
        rawData(i,:,:)  = x;  % for MUSIC
        Label(i).K      = sig.K;
        Label(i).theta  = sig.theta;
        Label(i).f_c    = sig.f_c;
        Label(i).a      = sig.P;
        Label(i).N      = sig.N;
        Label(i).SNRdB  = sig.SNRdB;
        
        if strcmp(sig.geometry,'VectorSensor')
            Label(i).phi    = sig.phi;
            Label(i).gamma  = sig.gamma;
            Label(i).ita    = sig.ita;
            Label(i).dt     = sig.dt;
            Label(i).deltaT = sig.deltaT;
        end
        
        if mod(i,1000) == 0
            processMsg = sprintf('data generating %.2f %%', i*100.0/Sizefull);
            disp(processMsg);
            toc;
        end
    end
end
toc;

% tagTargetLabel = strcat(sig.caseID,'TargetLabel.mat');
% tagSampTarget  = strcat('samp',sig.caseID,'Target.mat');
% tagSampInput   = strcat('samp',sig.caseID,'Input.mat');
for i = 1:3
    permID = randperm(Sizefull);
    allData = allData(permID,:);
    Label = Label(permID);
end

save( strcat('./Data/',sig.caseID,'Data.mat'),    'allData');
save( strcat('./Data/',sig.caseID,'RawData.mat'), 'rawData');
save( strcat('./Data/',sig.caseID,'Label.mat'),   'Label');

end