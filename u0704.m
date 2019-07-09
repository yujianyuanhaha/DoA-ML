% given values
pos = [8 4 ;    % startpoint
       2 7 ] ;  % endpoint
nturns = 3 ;    % number of turns (integer value)
% engine
dp = diff(pos,1,1) ;
R = hypot(dp(1), dp(2)) ;
phi0 = atan2(dp(2), dp(1)) ;
phi = linspace(0, nturns*2*pi, 10000) ; % 10000 = resolution
r = linspace(0, R, numel(phi)) ;
x = pos(1,1) + r .* cos(phi + phi0) ;
y = pos(1,2) + r  .* sin(phi + phi0) ;
plot(x,y,'b-',pos(:,1),pos(:,2),'ro-') ; % nturns crossings, including end point