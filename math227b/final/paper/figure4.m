function lander_fig4_repro()
    % --- Constants ---
    D_prime = 10;        % Diffusion (um^2/s)
    k_deg = 2e-4;       % Degradation (s^-1)
    x_max = 100;        % Domain size (microns)
    m = 0;              % Slab symmetry
    x = linspace(0, x_max, 200);

    % Scenarios from Figure 4 caption: [v_Rtot, kon_Rtot, koff, dt, t_max, ylim]
    % v_Rtot is treated as the flux value entering at x=0
    scenarios = {
        [5e-4, 1.32, 1e-6, 300, 3600, 1.0],   % Panel A
        [5e-4, 0.01, 1e-6, 600, 5400, 1.0],   % Panel B
        [5e-5, 1.32, 1e-6, 1800, 14400, 0.25], % Panel C
        [5e-5, 0.01, 1e-6, 1800, 14400, 0.25]  % Panel D
    };

    figure('Color', 'w', 'Name', 'Lander Fig 4 Reproduction');

    for i = 1:4
        s = scenarios{i};
        v_in = s(1); konR = s(2); koff = s(3); dt = s(4); t_max = s(5); y_lim = s(6);

        t = 0:dt:t_max;

        % Solve PDE
        % sol(:,:,1) is A (free), sol(:,:,2) is B (bound)
        sol = pdepe(m, @(x,t,u,du) lander_pde(x,t,u,du, D_prime, konR, koff, k_deg), ...
                       @(x) [0; 0], ...
                       @(xl,ul,xr,ur,t) lander_bc(xl,ul,xr,ur,t, v_in, D_prime), ...
                       x, t);

        B = sol(:,:,2);

        subplot(2,2,i);
        % Plot all time steps except t=0 (which is an empty line)
        plot(x, B(2:end,:)', 'k', 'LineWidth', 1.1);
        
        title(['Panel ', char(64+i)]);
        xlabel('distance (\mum)'); ylabel('bound / R_{tot}');
        xlim([0, 100]); ylim([0, y_lim]);
        grid on; set(gca, 'Box', 'off');
    end
end

% --- PDE Equation: Equations 1 and 2' ---
function [c,f,s] = lander_pde(x,t,u,du,D,konR,koff,kdeg)
    A = u(1); B = u(2);
    c = [1; 1];
    f = [D * du(1); 0]; % Only free ligand (A) diffuses
    
    % Net binding rate
    binding = konR * A * (1 - B) - koff * B;
    
    s = [-binding;            % Change in Free Ligand
          binding - kdeg * B]; % Change in Bound Complex
end

% --- Boundary Conditions ---
function [pl,ql,pr,qr] = lander_bc(xl,ul,xr,ur,t,v_in,D)
    % Left Boundary (x=0): Incoming Flux
    % pdepe flux is f = D * du/dx. 
    % We want -D * du/dx = v_in  => f + v_in = 0
    pl = [v_in; 0]; 
    ql = [1; 1]; 
    
    % Right Boundary (x=100): No Flux
    pr = [0; 0];
    qr = [1; 1];
end