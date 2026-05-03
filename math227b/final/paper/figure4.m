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
    configs = {
        % Panel A: High Affinity (Sigmoidal "Traveling Wave")
        struct('v_val', 5.5e-4, 'kon_eff', 1.32, 'koff', 1e-6, 'dt', 300,  'max_t', 3600,  'ymax', 1.0),  
        
        % Panel B: Low Affinity (Exponential-like "Filling" profile)
        struct('v_val', 2.0e-3, 'kon_eff', 0.0012, 'koff', 1e-6, 'dt', 600,  'max_t', 5400,  'ymax', 1.0),  
        
        % Panel C: 
        struct('v_val', 5e-6, 'kon_eff', 1.32, 'koff', 1e-6, 'dt', 1800, 'max_t', 10800, 'ymax', 0.25), 
        
        % Panel D: 
        struct('v_val', 5e-5, 'kon_eff', 0.01, 'koff', 1e-6, 'dt', 1800, 'max_t', 10800, 'ymax', 0.25)  
    };
    
    fig_labels = {'A', 'B', 'C', 'D'};
    figure('Color', 'w', 'Position', [100 100 950 750]);
    
    % Define a global colormap based on the maximum simulation time (3 hours)
    % so that colors are consistent across all subplots.
    max_global_hours = 3.0; 
    cmap_base = parula(256); 
    
    % Specific hours to include in the legend
    hours_to_label = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0];
    
    for i = 1:4
        cfg = configs{i};
        t = 0:cfg.dt:cfg.max_t;
        t_hours = t / 3600;
        v_flux = cfg.v_val * L_scale; % Applied scaling
        
        % Solve PDE
        sol = pdepe(0, @(x,t,u,dudx) pdefun(x,t,u,dudx, D, cfg.kon_eff, cfg.koff, k_deg), ...
                       @icfun, ...
                       @(xl,ul,xr,ur,t) bcfun(xl,ul,xr,ur,t, D, v_flux), ...
                       x, t);
        
        B = sol(:,:,2); % Bound fraction
        
        subplot(2, 2, i);
        hold on;
        
        h_lines = [];
        h_labels = {};
        
        % Plot each time step with a color corresponding to its time
        for j = 1:length(t_hours)
            % Map current time to a colormap index [1, 256]
            c_idx = max(1, min(256, round((t_hours(j) / max_global_hours) * 255) + 1));
            line_color = cmap_base(c_idx, :);
            
            p = plot(x, B(j, :), 'Color', line_color, 'LineWidth', 1.5);
            
            % Check if this time step is in our list of hours to label
            [err, ~] = min(abs(hours_to_label - t_hours(j)));
            if err < 1e-3
                h_lines(end+1) = p;
                h_labels{end+1} = sprintf('%.2g h', t_hours(j));
            end
        end
        
        % Aesthetics
        set(gca, 'TickDir', 'out', 'Box', 'off', 'FontSize', 11, 'LineWidth', 1);
        title(fig_labels{i}, 'FontWeight', 'bold', 'FontSize', 14, 'Units', 'normalized', 'Position', [-0.1, 1.05]);
        xlabel('Distance (\mum)', 'FontSize', 12);
        ylabel('Bound / R_{tot}', 'FontSize', 12);
        xlim([0 100]);
        ylim([0 cfg.ymax]);
        
        % Add publication-style legend
        if ~isempty(h_lines)
            legend(h_lines, h_labels, 'Location', 'northeast', 'Box', 'off', 'FontSize', 10);
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
