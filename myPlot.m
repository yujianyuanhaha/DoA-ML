function myPlot(eerMdeg,RMSE,errors,ground,predict, sig, nn, Case, net, tr)


figure;
histogram(errors);
clear title xlabel ylabel;
title('estimate error histogram');
xlabel('estimate error (rad)');
ylabel('count');
grid on;
saveas(gcf,'./Fig/hist.png');


%cdf of error
figure;
[hy, hx] = ecdf(errors);
plot(hx,hy);
clear title xlabel ylabel;
title('CDF of error');
xlabel('estimate error (rad)');
ylabel('sig.P(X < x)');
grid on;
saveas(gcf,'cdf.png');
% save(strcat(sig.caseID,'hx.mat'),'hx');
% save(strcat(sig.caseID,'hy.mat'),'hy');

figure;
scatter(ground,predict);
clear title xlabel ylabel;
grid on;
title('ground vs ML estimate')
xlabel('estimate ML');
ylabel('ground');
saveas(gcf,'./Fig/scatter.png');

%neural network topoloy
jframe = view(net);
hFig = figure('Menubar','none', 'Position',[100 100 1200 166]);
jpanel = get(jframe,'ContentPane');
[~,h] = javacomponent(jpanel);
set(h, 'units','normalized', 'position',[0 0 1 1])
jframe.setVisible(false);
jframe.dispose();
set(hFig, 'PaperPositionMode', 'auto')
saveas(hFig, './Fig/nn.png')
close(hFig)

figure;
plotperform(tr);
saveas(gcf,'./Fig/perf.png');

figure;
plottrainstate(tr);
saveas(gcf,'./Fig/trainState.png');

figure;
ploterrhist(errors);
saveas(gcf,'./Fig/hist2.png');

figure;
plotregression(ground, predict);
saveas(gcf,'./Fig/regression.png');

emailTitle = strcat('(Dell) ',sig.geometry, ' ',Case.note, sprintf(' mean error %.4f deg, RMSE %.4f deg',eerMdeg,RMSE));
setting = sprintf('sig.SNRdB=%.2f; P1=%.2f, P2=%.2f, P3=%.2f, P4=%.2f, P5=%.2f, ',...
    sig.SNRdB,sig.P(1),sig.P(2),sig.P(3),sig.P(4),sig.P(5));
nnSetting = sprintf('time = %.2f, trainFcn = %s, nn.isRad = %sig.d, nn.isBayes = %sig.d ,%s', ...
    tr.time(end),tr.trainFcn,nn.isRad, nn.isBayes,sig.type);
content = strcat( setting,nnSetting);
% attachment = {'hist.png','cdf.png','scatter.png','nn.png',...
%     'perf.png','trainState.png','hist2.png',...
%     'regression.png',strcat(sig.caseID,'errors.mat'),strcat(sig.caseID,'ground.mat'),...
%     strcat(sig.caseID,'predict.mat'),strcat(sig.caseID,'hx.mat'),strcat(sig.caseID,'hy.mat')};
 attachment = {'./Fig/scatter.png',strcat('./Data',sig.caseID,'errors.mat'),...
     strcat('./Data',sig.caseID,'ground.mat'),strcat('./Data',sig.caseID,'predict.mat')};
if Case.isSend
    sendEmail(emailTitle,content,attachment);
end

end

