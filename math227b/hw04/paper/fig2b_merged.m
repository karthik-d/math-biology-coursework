function breast_cancer_fig1_structured
    %% =============================================================
    %% Modular DDE Simulation for Breast Cancer Models
    %% =============================================================
    % Define simulation settings
    tau = 1.2;           % Delay (days)
    tspan = [0 1200];   % Simulation time
    % Initial condition: [CSC; PC; TDC]
    total_init = 1e5;
    % perc_csc = 0.015;
    % perc_pc = 0.259;
    % perc_tdc = 0.726;
    perc_csc = 0.8;
    perc_pc = 0.01;
    perc_tdc = 0.19;
    % perc_csc = 1;
    % perc_pc = 0.0;
    % perc_tdc = 0.0;
    history = total_init*[perc_csc; perc_pc; perc_tdc]; 

    % --- Define parameter sets for each model ---
    % Combined Model (Type I + Type II)
    params_combined = struct(...
        'p0', 0.5, 'q0', 0.2, ...
        'p1', 0.5, 'q1', 0.1, ...
        'v0', 1.0, 'v1', 2.0, ...
        'd0', 0.01, 'd1', 0.05, 'd2', 0.1, ...
        'gamma01', 2e-17, 'gamma02', 6e-15, ...
        'gamma11', 1e-15, 'gamma12', 2e-14, ...
        'beta0', 7e-18, 'beta1', 3e-18);

    % --- Store models in a struct array ---
    models = {...
        struct('name','Combined','func',@dde_combined,'params',params_combined,'color','r')};

    %% =============================================================
    %% Run simulations and collect results
    %% =============================================================
    results = cell(size(models));
    figure('Color','w'); hold on; grid on;
    xlabel('Time (days)'); ylabel('CD44+CD24- (%)'); title('CSC Proportion over Time');
    
    for i = 1:numel(models)
        m = models{i};
        sol = dde23(@(t,y,Z) m.func(t,y,Z,m.params), tau, history, tspan);
        total_cells = sum(sol.y,1);
        plot(sol.x, sol.y(1, :)./total_cells*100, m.color,'LineWidth',2);
        results{i}.sol = sol;
        results{i}.total_cells = total_cells;
        results{i}.name = m.name;
    end
    
   observations = [
         0, 79;
         2, 15;
         4, 37;
         6, 40;
         8, 46;
        10, 38;
        12, 16;
        14, 13
    ];
    
    t_data = observations(:, 1);
    y_data = observations(:, 2); % This is in %
    plot(t_data, y_data, 'ko');

    legend('Given DDE Parameters', 'Observed');
    axis([0 15 0 100]);
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

