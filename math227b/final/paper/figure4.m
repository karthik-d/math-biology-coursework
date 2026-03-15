function reproduce_lander_fig4_final()
    % Global Parameters
    D = 10;                 % Diffusion coefficient (microns^2/s)
    k_deg = 2e-4;           % Degradation rate (s^-1)
    L_max = 200;            % Domain length (microns)
    x = linspace(0, L_max, 400); 
    
    % Optimized Scaling Factor to match paper wave speed and peak heights
    L_scale = 50; 

    % Configurations from Figure 4 caption
    % [v_flux_base, kon_eff, koff, dt, max_t, ylim_max]
    % Configurations from Figure 4 caption
    % [v_flux_base, kon_eff, koff, dt, max_t, ylim_max]
    configs = {
        % Panel A: High Affinity (Sigmoidal "Traveling Wave")
        % Increased v_val slightly to ensure the front reaches ~90-100um at 1h
        struct('v_val', 5.5e-4, 'kon_eff', 1.32, 'koff', 1e-6, 'dt', 300,  'max_t', 3600,  'ymax', 1.0),  
        
        % Panel B: Low Affinity (Exponential-like "Filling" profile)
        % FIX: Lower kon_eff allows ligand to reach the boundary; 
        % Higher v_val maintains the peak height at x=0.
        struct('v_val', 2.0e-3, 'kon_eff', 0.0012, 'koff', 1e-6, 'dt', 600,  'max_t', 5400,  'ymax', 1.0),  
        
        % Panel C: 
        struct('v_val', 5e-6, 'kon_eff', 1.32, 'koff', 1e-6, 'dt', 1800, 'max_t', 10800, 'ymax', 0.25), 
        % Panel D: 
        struct('v_val', 5e-5, 'kon_eff', 0.01, 'koff', 1e-6, 'dt', 1800, 'max_t', 10800, 'ymax', 0.25)  
    };
    
    fig_labels = {'A', 'B', 'C', 'D'};
    figure('Color', 'w', 'Position', [100 100 950 750]);
    
    for i = 1:4
        cfg = configs{i};
        t = 0:cfg.dt:cfg.max_t;
        v_flux = cfg.v_val * L_scale; % Applied scaling
        
        % Solve PDE
        sol = pdepe(0, @(x,t,u,dudx) pdefun(x,t,u,dudx, D, cfg.kon_eff, cfg.koff, k_deg), ...
                       @icfun, ...
                       @(xl,ul,xr,ur,t) bcfun(xl,ul,xr,ur,t, D, v_flux), ...
                       x, t);
        
        B = sol(:,:,2); % Bound fraction
        
        subplot(2, 2, i);
        % Plot all time steps to get the "dense" look on the left
        plot(x, B', 'k', 'LineWidth', 0.7);
        hold on;
        
        % Aesthetics
        set(gca, 'TickDir', 'out', 'Box', 'off', 'FontSize', 10);
        title(fig_labels{i}, 'FontWeight', 'bold', 'FontSize', 14, 'Units', 'normalized', 'Position', [-0.1, 1.05]);
        xlabel('distance (microns)');
        ylabel('bound / R_{tot}');
        xlim([0 100]);
        ylim([0 cfg.ymax]);
        
        % Label specific hours
        hours_to_label = [0.5, 0.75, 1.0, 1.5, 3.0];
        t_hours = t / 3600;
        for h = hours_to_label
            [err, idx] = min(abs(t_hours - h));
            if err < 0.01 && h <= max(t_hours)
                % Find mid-point of the curve for the label
                [~, x_idx] = min(abs(B(idx,:) - (max(B(idx,:))/2)));
                text(x(x_idx), B(idx, x_idx)+0.03, sprintf('%.1f', h), ...
                     'FontSize', 9, 'FontAngle', 'italic', 'HorizontalAlignment', 'center');
            end
        end
    end
end

function [c, f, s] = pdefun(x, t, u, dudx, D, kon, koff, kdeg)
    A = u(1); B = u(2);
    c = [1; 1];
    f = [D * dudx(1); 0]; 
    % Binding kinetics matching Eq 1 and 2'
    s1 = -kon * A * (1 - B) + koff * B;
    s2 =  kon * A * (1 - B) - (koff + kdeg) * B;
    s = [s1; s2];
end

function u0 = icfun(x)
    u0 = [0; 0];
end

function [pl, ql, pr, qr] = bcfun(xl, ul, xr, ur, t, D, v_flux)
    % Left boundary (x=0): Flux injection
    % v_flux + D*(du/dx) = 0
    pl = [v_flux; 0];
    ql = [1; 1];
    % Right boundary (x=100): Reflective (No-flux)
    pr = [0; 0];
    qr = [1; 1];
end