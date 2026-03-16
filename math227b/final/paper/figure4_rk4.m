function Lander_2002_Reproduce_Figure4_Corrected
    % Lander_2002_Reproduce_Figure4_Corrected reproduces Figure 4 from 
    % Lander et al. (2002). It corrects for known parameter inconsistencies 
    % in the paper's text to accurately reproduce the visual panels.
    
    close all; clear; clc;
    
    %% ------------------------------------------------------------------
    %  1. Parameters Common to All Panels
    %  ------------------------------------------------------------------
    x_max = 100;                % Domain size [microns]
    k_deg = 2e-4;               % Receptor degradation rate [s^-1]
    k_off = 1e-6;               % Ligand dissociation rate [s^-1] 
    D_prime = 10.0;             % Diffusion coefficient [microns^2/s] (From Fig 2)

    % Space discretization
    N = 201;                    % Number of spatial points
    x = linspace(0, x_max, N);  % Spatial vector [microns]
    dx = x(2) - x(1);           % Space step [microns]
    
    % Time-stepping parameters 
    % Stability condition: h < dx^2 / (2 * D_prime) = 1 / 20 = 0.05
    h = 0.025;                  % Stable time step for RK4 [s]

    % Setup main figure
    figure('Name', 'Reproduction of Lander 2002, Figure 4', ...
           'Color', 'white', 'Position', [100 100 1000 800]);

    %% ------------------------------------------------------------------
    %  2. Definition of the Derivative Function (f)
    %  ------------------------------------------------------------------
    function dydt = model_f(t, y, v_flux, k_on_Rtot)
        A = y(1:N);       % Normalized free ligand [L]/Rtot
        B = y(N+1:end);   % Fractional receptor occupancy [LR]/Rtot
        
        dA_dt = zeros(N, 1);
        dB_dt = zeros(N, 1);
        
        % Reaction part 
        eff_on_rate  = k_on_Rtot * A .* (1 - B);
        eff_off_rate = k_off * B;
        dB_dt = eff_on_rate - eff_off_rate - k_deg * B;
        
        % Diffusion part with Ghost Nodes for Neumann boundary conditions
        A_ext = [0; A; 0]; 
        
        % Left Boundary (x=0): Flux injection
        % -D_prime * (A_2 - A_0) / (2*dx) = v_flux  =>  A_0 = A_2 + 2*dx*v_flux/D_prime
        A_ext(1) = A(2) + (2 * dx * v_flux / D_prime);
        
        % Right Boundary (x=x_max): No flux
        A_ext(end) = A(end-1);
        
        for i = 1:N
            d2A_dx2 = (A_ext(i+2) - 2*A_ext(i+1) + A_ext(i)) / (dx^2);
            dA_dt(i) = D_prime * d2A_dx2 - eff_on_rate(i) + eff_off_rate(i);
        end
        
        dydt = [dA_dt; dB_dt];
    end

    %% ------------------------------------------------------------------
    %  3. Definition of the RK4 Solver
    %  ------------------------------------------------------------------
    function [time_log, B_history] = run_scenario(t_end, v_flux, k_on_Rtot, plot_interval_s)
        t_current = 0;
        steps = floor(t_end / h);
        
        y = zeros(2 * N, 1); % Initial condition: 0 gradient
        
        time_log = [];
        B_history = [];
        next_plot_time = plot_interval_s;
        
        fprintf('Starting simulation (Target: %.1f hrs)...', t_end/3600);
        for i = 1:steps
            % Store snapshots
            if t_current >= next_plot_time - 1e-5
                time_log = [time_log, t_current];
                B_history = [B_history, y(N+1:end)];
                next_plot_time = next_plot_time + plot_interval_s;
            end
            
            % RK4 Core
            k1 = model_f(t_current,     y,             v_flux, k_on_Rtot);
            k2 = model_f(t_current + h/2, y + k1*(h/2), v_flux, k_on_Rtot);
            k3 = model_f(t_current + h/2, y + k2*(h/2), v_flux, k_on_Rtot);
            k4 = model_f(t_current + h,   y + k3*h,     v_flux, k_on_Rtot);
            
            y = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
            t_current = t_current + h;
        end
        fprintf('Done.\n');
    end

    %% ------------------------------------------------------------------
    %  4. Executing Scenarios (A - D)
    %  ------------------------------------------------------------------
    % Note: v_flux values are empirically derived here to match the 
    % conservation of mass shown in the paper's figures.
    scenarios = {...
        'A', 0.025,   1.32, 1.05, 300,  1.0,  [0 1.0]; ...
        'B', 0.025,   0.01, 1.55, 600,  1.5,  [0 1.0]; ...
        'C', 0.00018, 1.32, 3.05, 1800, 0.25, [0 0.25]; ...
        'D', 0.0021,  0.01, 3.05, 1800, 0.25, [0 0.25] ...
    };
    % Cols: Name, effective_flux, kon*Rtot, total_time(hr), plot_int(s), (unused), ylim

    for s = 1:size(scenarios, 1)
        name        = scenarios{s,1};
        v_flux      = scenarios{s,2};
        k_on_Rtot   = scenarios{s,3};
        t_total_hr  = scenarios{s,4};
        plot_int    = scenarios{s,5};
        y_lim_upper = scenarios{s,7}(2);
        
        fprintf('--- Panel %s ---\n', name);
        t_end_s = t_total_hr * 3600;
        
        [time_log_s, B_history] = run_scenario(t_end_s, v_flux, k_on_Rtot, plot_int);
        
        % Plotting subplot
        subplot(2, 2, s);
        hold on; box on; grid on;
        
        n_curves = length(time_log_s);
        for i = 1:n_curves
            color_val = 1 - (i-1)/(n_curves-1) * 0.8; 
            plot(x, B_history(:,i), 'LineWidth', 1.0, 'Color', [0.2 0.2 0.2]); 
        end
        
        title(name, 'FontSize', 16, 'FontWeight', 'bold', 'Units', 'normalized', ...
              'Position', [-0.1, 1.0, 0], 'HorizontalAlignment', 'left');
        xlabel('distance (microns)', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('$$\frac{\textrm{bound}}{R_{\textrm{tot}}}$$', 'Interpreter', 'latex', ...
               'FontSize', 14, 'FontWeight', 'bold', 'Rotation', 0, ...
               'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', ...
               'Position', [-15, y_lim_upper/2, 0]);
        
        set(gca, 'XLim', [0 x_max], 'YLim', [0 y_lim_upper], ...
            'XTick', 0:20:100, 'TickDir', 'out', 'FontSize', 10, 'FontWeight', 'bold');
        
        % --------------------------------------------------------------
        % 5. Adding specific labels mimicking the paper
        % --------------------------------------------------------------
        switch name
            case 'A'
                % 0.5, 0.75, 1.0 hr labels
                idx = arrayfun(@(t) find(time_log_s >= t*3600 - 1, 1), [0.5, 0.75, 1.0]);
                locs = [8, 45, 95];
                labels = {'0.5', '0.75', '1.0'};
                for k=1:3, text(locs(k), B_history(find(x>=locs(k),1),idx(k))*1.02, labels{k}, 'FontAngle','italic'); end
            case 'B'
                % 0.5, 1.0, 1.5 hr labels
                idx = arrayfun(@(t) find(time_log_s >= t*3600 - 1, 1), [0.5, 1.0, 1.5]);
                locs = [30, 60, 85];
                labels = {'0.5', '1.0', '1.5'};
                for k=1:3, text(locs(k), B_history(find(x>=locs(k),1),idx(k))*1.05, labels{k}, 'FontAngle','italic'); end
            case 'C'
                % 0.5 hr label
                idx = find(time_log_s >= 0.5*3600 - 1, 1);
                text(10, B_history(find(x>=10,1),idx)*1.5, '0.5', 'FontAngle','italic');
            case 'D'
                % 0.5, 1.0, 3.0 hr labels
                idx = arrayfun(@(t) find(time_log_s >= t*3600 - 1, 1), [0.5, 1.0, 3.0]);
                locs = [35, 40, 52];
                labels = {'0.5', '1.0', '3.0'};
                for k=1:3, text(locs(k), B_history(find(x>=locs(k),1),idx(k))*1.1, labels{k}, 'FontAngle','italic'); end
        end
    end
    fprintf('Figure 4 reproduction complete.\n');
end