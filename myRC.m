function p = myRC(beta,span,sps,shape)
% replacement of p     = rcosdesign(beta,span,sps,shape)


% ref  value of official lib
% p = [0.0266468801359843,-0.0276231535739088,-0.0855374118147732, ...
%     -0.0998121297879238,-0.0322651579086280,0.119473681399052,...
%     0.312317558188603,0.473734829004615,0.536592673759239,...
%     0.473734829004615,0.312317558188603,0.119473681399052,...
%     -0.0322651579086280,-0.0998121297879238,-0.0855374118147732,...
%     -0.0276231535739088,0.0266468801359843];


offset = span*sps/2;
T = span;
dt = T/sps;

p = zeros(1,span*sps+1);

for i = - offset:1:offset
    if abs(i*dt) == T/(2*beta)
        p(i+offset+1) = pi/(4*T)*sinc(1/(2*beta));
    else
        p(i+offset+1) = 1/T * sinc(i*dt/T) * cos(pi*beta*i*dt/T) ...
            /(1-(2*beta*i*dt/T)^2);
    end
    
end

end


