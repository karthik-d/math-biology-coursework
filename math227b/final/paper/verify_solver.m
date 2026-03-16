function reproduce_crampin_front_heatmap()
% Reproduces pattern formation behind a travelling wave front 
% for the 1D Schnakenberg model and plots as a spatiotemporal heatmap.

    % Reaction parameters
    a = 0.1;
    b = 0.9;
    Du = 1.0;
    Dv = 0.01;
    
    % Domain and time
    L_max = 40.0;
    x = linspace(0, L_max, 400);
    % Use many time points for a smooth heatmap
    t = linspace(0, 300, 300); 
    
    % Solve PDE using pdepe framework
    sol = pdepe(0, @(x,t,u,dudx) pdefun(x,t,u,dudx, a,b,Du,Dv), ...
                   @icfun, ...
                   @bcfun, ...
                   x, t);
                   
    v = sol(:,:,2); % Extract the activator (v)
    
    % Plot the spatiotemporal heatmap
    figure('Color', 'w', 'Position', [100 100 700 550]);
    
    % imagesc is standard for heatmaps in MATLAB.
    % We set 'YDir' to 'normal' so t=0 is at the bottom.
    imagesc(x, t, v);
    set(gca, 'YDir', 'normal'); 
    
    % Aesthetics
    colormap('parula');
    c = colorbar;
    c.Label.String = 'Activator Concentration (v)';
    c.Label.FontSize = 12;
    
    set(gca, 'TickDir', 'out', 'Box', 'off', 'FontSize', 12, 'LineWidth', 1);
    title('Spatiotemporal Pattern Formation (Schnakenberg)', 'FontSize', 14);
    xlabel('Distance x', 'FontSize', 12);
    ylabel('Time t', 'FontSize', 12);
    ylim([0 60])
end

function [c, f, s] = pdefun(x, t, u, dudx, a, b, Du, Dv)
    c = [1; 1];
    f = [Du; Dv] .* dudx; 
    % Schnakenberg kinetics
    s1 = b - u(1)*(u(2)^2);
    s2 = a + u(1)*(u(2)^2) - u(2);
    s = [s1; s2];
end

function u0 = icfun(x)
    % Homogeneous steady state with a localized perturbation at center
    a = 0.1; b = 0.9;
    u_ss = b / (a+b)^2; % 0.9
    v_ss = a + b;       % 1.0
    
    % Trigger the front from the center (x=20)
    u0 = [u_ss; 
          v_ss + 0.5 * exp(-(x-20)^2 / 2.0)];
end

function [pl, ql, pr, qr] = bcfun(xl, ul, xr, ur, t)
    % Zero-flux boundaries
    pl = [0; 0];
    ql = [1; 1];
    pr = [0; 0];
    qr = [1; 1];
end
