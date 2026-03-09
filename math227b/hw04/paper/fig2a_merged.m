function breast_cancer_fig1_structured
    %% =============================================================
    %% Modular DDE Simulation for Breast Cancer Models
    %% =============================================================
    % Define simulation settings
    tau = 2;           % Delay (days)
    tspan = [0 1200];   % Simulation time
    % Initial condition: [CSC; PC; TDC]
    total_init = 1e7;
    perc_csc = 0.015;
    perc_pc = 0.259;
    perc_tdc = 0.726;
    history = total_init*[perc_csc; perc_pc; perc_tdc]; 

    % --- Define parameter sets for each model ---
    % Combined Model (Type I + Type II)
    params_combined = struct(...
        'p0', 0.5, 'q0', 0.2, ...
        'p1', 0.5, 'q1', 0.1, ...
        'v0', 1.0, 'v1', 2.0, ...
        'd0', 0.01, 'd1', 0.05, 'd2', 0.1, ...
        'gamma01', 1e-14, 'gamma02', 1e-16, ...
        'gamma11', 1e-13, 'gamma12', 1e-15, ...
        'beta0', 8e-12, 'beta1', 4e-13);

    % Type II Feedback Model
   params_typeII = struct(...
        'p0', 0.5, 'q0', 0.2, ...
        'p1', 0.5, 'q1', 0.1, ...
        'v0', 1.0, 'v1', 2.0, ...
        'd0', 0.01, 'd1', 0.05, 'd2', 0.1, ...
        'gamma01', 5e-14, 'gamma02', 7e-15, ...
        'gamma11', 6e-13, 'gamma12', 2e-15);

    % Type I Feedback Model
    params_typeI = struct(...
        'p0', 0.5, 'q0', 0.2, ...
        'p1', 0.5, 'q1', 0.1, ...
        'v0', 1.0, 'v1', 2.0, ...
        'd0', 0.01, 'd1', 0.05, 'd2', 0.1, ...
        'beta0', 2e-11, 'beta1', 3e-12);

    % Basic Model (no feedback)
    params_basic = struct(...
        'p0', 0.25, 'q0', 0.2, ...
        'p1', 0.3, 'q1', 0.1, ...
        'v0', 1.0, 'v1', 2.0, ...
        'd0', 0.01, 'd1', 0.05, 'd2', 0.1);

    % --- Store models in a struct array ---
    models = {...
        struct('name','Combined','func',@dde_combined,'params',params_combined,'color','r'), ...
        struct('name','Type II','func',@dde_typeII,'params',params_typeII,'color','g'), ...
        struct('name','Type I','func',@dde_typeI,'params',params_typeI,'color','b'), ...
        struct('name','Basic','func',@dde_basic,'params',params_basic,'color','k--')};

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
        % plot(sol.x, total_cells, m.color,'LineWidth',2);
        plot(sol.x, sol.y(1, :)./total_cells*100, m.color,'LineWidth',2);
        results{i}.sol = sol;
        results{i}.total_cells = total_cells;
        results{i}.name = m.name;
    end
    
    % % plot data points.
    % observations = [ ...
    %     0.0,  0.20;
    %     1.0,  0.10;
    %     2.0,  0.14;
    %     3.0,  0.23;
    %     4.0,  0.27;
    %     5.0,  0.28;
    %     6.0,  0.47;
    %     7.0,  0.70;
    %     8.0,  1.12;
    %     9.0,  1.57;
    %    10.0,  2.15;
    %    12.0,  3.00;
    %    13.0,  3.05;
    %    14.0,  3.02];
    % t_obs = observations(:, 1); 
    % y_obs = observations(:, 2)*1e6;
    % plot(t_obs, y_obs, 'ko');
    % 
    model_names = cellfun(@(m) m.name, models, 'UniformOutput', false);
    legend(model_names, 'Location','northwest');
    axis([0 20 0 8]);

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

function dydt = dde_typeII(t,y,Z,p)
    % Type II feedback only
    x0 = y(1); x1 = y(2); x2 = y(3);
    x2_delayed = Z(3,1);

    p0_eff = p.p0 / (1 + p.gamma01 * x2_delayed^2);
    q0_eff = p.q0 / (1 + p.gamma02 * x2_delayed^2);
    p1_eff = p.p1 / (1 + p.gamma11 * x2_delayed^2);
    q1_eff = p.q1 / (1 + p.gamma12 * x2_delayed^2);

    dx0 = (p0_eff - q0_eff)*p.v0*x0 - p.d0*x0;
    dx1 = (1 - p0_eff + q0_eff)*p.v0*x0 + (p1_eff - q1_eff)*p.v1*x1 - p.d1*x1;
    dx2 = (1 - p1_eff + q1_eff)*p.v1*x1 - p.d2*x2;
    dydt = [dx0; dx1; dx2];
end

function dydt = dde_typeI(t,y,Z,p)
    % Type I feedback only
    x0 = y(1); x1 = y(2); x2 = y(3);
    x2_delayed = Z(3,1);

    v0_eff = p.v0 / (1 + p.beta0 * x2_delayed^2);
    v1_eff = p.v1 / (1 + p.beta1 * x2_delayed^2);

    dx0 = (p.p0 - p.q0)*v0_eff*x0 - p.d0*x0;
    dx1 = (1 - p.p0 + p.q0)*v0_eff*x0 + (p.p1 - p.q1)*v1_eff*x1 - p.d1*x1;
    dx2 = (1 - p.p1 + p.q1)*v1_eff*x1 - p.d2*x2;
    dydt = [dx0; dx1; dx2];
end

function dydt = dde_basic(t,y,~,p)
    % No feedback
    x0 = y(1); x1 = y(2); x2 = y(3);

    dx0 = (p.p0 - p.q0)*p.v0*x0 - p.d0*x0;
    dx1 = (1 - p.p0 + p.q0)*p.v0*x0 + (p.p1 - p.q1)*p.v1*x1 - p.d1*x1;
    dx2 = (1 - p.p1 + p.q1)*p.v1*x1 - p.d2*x2;
    dydt = [dx0; dx1; dx2];
end