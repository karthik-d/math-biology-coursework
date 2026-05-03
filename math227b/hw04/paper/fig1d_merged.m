function breast_cancer_fig1_structured
    %% =============================================================
    %% Modular DDE Simulation for Breast Cancer Models
    %% =============================================================
    % Define simulation settings
    tau = 1.2;           % Delay (days)
    tspan = [0 1200];   % Simulation time
    % Initial condition: [CSC; PC; TDC]
    total_init = 1e4;
    perc_csc = 0.015;
    perc_pc = 0.259;
    perc_tdc = 0.726;
    % perc_csc = 0.8;
    % perc_pc = 0.1;
    % perc_tdc = 0.1;
    history = total_init*[perc_csc; perc_pc; perc_tdc]; 

    % --- Define parameter sets for each model ---
    % Combined Model (Type I + Type II)
    params_combined = struct(...
        'p0', 0.5, 'q0', 0.2, ...
        'p1', 0.5, 'q1', 0.1, ...
        'v0', 1.0, 'v1', 2.0, ...
        'd0', 0.01, 'd1', 0.05, 'd2', 0.1, ...
        'gamma01', 1e-23, 'gamma02', 2e-24, ...
        'gamma11', 4e-22, 'gamma12', 5e-23, ...
        'beta0', 8e-27, 'beta1', 4e-27);

    % --- Store models in a struct array ---
    models = {...
        struct('name','Combined','func',@dde_combined,'params',params_combined,'color','r')};

    %% =============================================================
    %% Run simulations and collect results
    %% =============================================================
    results = cell(size(models));
    figure('Color','w'); hold on; grid on;
    xlabel('Time (days)'); ylabel('Total Cell Number'); title('Total Cell Number Over Time');
    
    for i = 1:numel(models)
        m = models{i};
        % if (m.name ~= "Combined")
        %     continue;
        % end
        sol = dde23(@(t,y,Z) m.func(t,y,Z,m.params), tau, history, tspan);
        total_cells = sum(sol.y,1);
        plot(sol.x, total_cells, m.color,'LineWidth',2);
        % plot(sol.x, sol.y(1, :)./total_cells*100, m.color,'LineWidth',2);
        results{i}.sol = sol;
        results{i}.total_cells = total_cells;
        results{i}.name = m.name;
    end
    
    %% 1. Data Matrix
    data_matrix = [
        53,  0.0,  NaN,  NaN,  NaN; 56,  NaN,  NaN,  NaN,  0.0;
        60,  0.0,  NaN,  NaN,  NaN; 63,  NaN,  NaN,  NaN,  0.0;
        67,  0.0,  NaN,  NaN,  NaN; 69,  NaN,  NaN,  0.0,  NaN;
        70,  NaN,  NaN,  NaN,  0.1; 73,  0.0,  NaN,  NaN,  NaN;
        74,  NaN,  0.1,  NaN,  NaN; 76,  NaN,  NaN,  0.0,  NaN;
        77,  NaN,  NaN,  NaN,  0.2; 80,  NaN,  0.3,  NaN,  NaN;
        83,  NaN,  NaN,  0.1,  NaN; 84,  NaN,  NaN,  NaN,  0.4;
        88,  NaN,  0.9,  NaN,  NaN; 90,  NaN,  NaN,  0.4,  NaN;
        91,  NaN,  NaN,  NaN,  0.6; 96,  NaN,  1.2,  NaN,  NaN;
        97,  1.3,  NaN,  0.8,  NaN; 98,  NaN,  NaN,  NaN,  1.2;
        102, 2.4,  1.3,  NaN,  NaN; 104, NaN,  NaN,  1.7,  1.6;
        109, 4.7,  3.2,  NaN,  NaN; 111, NaN,  NaN,  2.9,  2.3;
        115, 5.3,  4.4,  NaN,  NaN; 120, 4.9,  NaN,  NaN,  NaN
    ];

    for i = 2:5
        col = data_matrix(:, i);
        valid = ~isnan(col);
        if sum(valid) > 2
            t_data = data_matrix(valid, 1);
            t_start = t_data(1);
            t_data_norm = t_data - t_start;
            N_data = (col(valid) .* 1e11); 
            
            plot(t_data(:), N_data(:), '.', 'MarkerSize', 20);
        end
    end

    model_names = cellfun(@(m) m.name, models, 'UniformOutput', false);
    legend('Combined', 'H605 #1', 'H605 #2', 'MCF7/HER2 #1', 'MCF7/HER2 #2');
    axis([50 120 0 7e11]);
    prettyfig;

    %% =============================================================
    %% Plot CSC percentage over time (like Fig 1d / Fig 2)
    %% =============================================================
    % figure('Color','w'); hold on; grid on;
    % xlabel('Time (days)'); ylabel('Percentage of CSCs (%)'); title('CSC Fraction Over Time');
    % 
    % for i = 1:numel(results)
    %     sol = results{i}.sol;
    %     total_cells = sum(sol.y,1);
    %     csc_frac = 100 * sol.y(1,:) ./ total_cells; % CSC fraction
    %     plot(sol.x, csc_frac, models{i}.color, 'LineWidth',2);
    % end
    % legend(model_names, 'Location','northeast');
    % axis([0 120 0 7e11]);
end


%% =============================================================
%% --- DDE Model Functions ---
%% =============================================================

function dydt = dde_combined(t,y,Z,p)
    % Combined Type I + Type II Feedback
    x0 = y(1); x1 = y(2); x2 = y(3);
    x2_delayed = Z(3,1);

    % Type II feedback on symmetric division
    p0_eff = p.p0 / (1 + p.gamma01 * x2_delayed^2);
    q0_eff = p.q0 / (1 + p.gamma02 * x2_delayed^2);
    p1_eff = p.p1 / (1 + p.gamma11 * x2_delayed^2);
    q1_eff = p.q1 / (1 + p.gamma12 * x2_delayed^2);

    % Type I feedback on proliferation rates
    v0_eff = p.v0 / (1 + p.beta0 * x2_delayed^2);
    v1_eff = p.v1 / (1 + p.beta1 * x2_delayed^2);

    % DDE System
    dx0 = (p0_eff - q0_eff)*v0_eff*x0 - p.d0*x0;
    dx1 = (1 - p0_eff + q0_eff)*v0_eff*x0 + (p1_eff - q1_eff)*v1_eff*x1 - p.d1*x1;
    dx2 = (1 - p1_eff + q1_eff)*v1_eff*x1 - p.d2*x2;
    dydt = [dx0; dx1; dx2];
end

