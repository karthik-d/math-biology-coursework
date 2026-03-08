function results = fit_breast_cancer_models(t_data_full, N_data_full)
    %% 1. Data Subsetting (Day 0 to 13)
    fit_mask = t_data_full >= 0 & t_data_full <= 13;
    t_data = t_data_full(fit_mask);
    N_data = N_data_full(fit_mask);
    
    %% 2. Settings
    tau = 1.2;
    tspan_fit = [min(t_data) max(t_data)];
    N0 = N_data(1);
    
    options = optimoptions('lsqnonlin', ...
        'Display','iter', ...
        'MaxIterations',500, ...
        'FunctionTolerance',1e-9, ...
        'StepTolerance',1e-9);
    
    %% 3. Define Baseline Parameters & Model Structs
    p_base = struct('p0',0.5,'q0',0.2,'p1',0.5,'q1',0.1,...
                    'v0',1.0,'v1',2.0,...
                    'd0',0.01,'d1',0.05,'d2',0.1,...
                    'gamma01',1e-14,'gamma02',1e-16,...
                    'gamma11',1e-13,'gamma12',1e-15,...
                    'beta0',1e-11,'beta1',5e-13);
    
    models = { ...
        struct('name','Combined','func',@dde_combined,'color','r'), ...
        struct('name','TypeII',  'func',@dde_typeII,  'color','g'), ...
        struct('name','TypeI',   'func',@dde_typeI,   'color','b'), ...
        struct('name','Basic',   'func',@dde_basic,   'color','k')};
    
    %% 4. Fit Each Model with Decoupled Constraints
    results = struct;
    for i = 1:length(models)
        m = models{i};
        fprintf('\n--- Fitting Model: %s ---\n', m.name);
        
        % Decouple Initial Guesses and Bounds based on Model Type
        % theta = [f_csc, f_pc, log10_feedback]
        switch m.name
            case 'Combined'
                % Fitting [f_csc, f_pc, log10_gamma, log10_beta]
                theta0 = [0.05, 0.1, -14.0, -11.0];
                lb     = [0.00, 0.0, -22.0, -18.0];
                ub     = [1.00, 1.0,  -8.0,  -6.0];
            case 'TypeII'
                % Fitting [f_csc, f_pc, log10_gamma]
                theta0 = [0.05, 0.1, -14.0];
                lb     = [0.00, 0.0, -22.0];
                ub     = [1.00, 1.0,  -8.0];
            case 'TypeI'
                % Fitting [f_csc, f_pc, log10_beta]
                theta0 = [0.05, 0.1, -11.0];
                lb     = [0.00, 0.0, -18.0];
                ub     = [1.00, 1.0,  -6.0];
            case 'Basic'
                % Fitting only [f_csc, f_pc]
                theta0 = [0.05, 0.1];
                lb     = [0.00, 0.0];
                ub     = [1.00, 1.0];
        end
        
        objfun = @(theta) model_residual(theta, m.func, p_base, t_data, N_data, tspan_fit, tau, N0, m.name);
        theta_fit = lsqnonlin(objfun, theta0, lb, ub, options);
    
        results(i).theta  = theta_fit;
        results(i).name   = m.name;
        results(i).func   = m.func;
        results(i).params = p_base;
        results(i).color  = m.color;
    end
    
    %% 5. Plot Results
    figure; hold on; grid on;
    scatter(t_data, N_data, 50, 'k', 'filled', 'DisplayName', 'Data');
    tplot = linspace(0, 20, 400); 
    for i = 1:length(results)
        theta = results(i).theta;
        N_model = simulate_model(theta, results(i).func, results(i).params, tplot, [0 20], tau, N0, results(i).name);
        plot(tplot, N_model, 'Color', results(i).color, 'LineWidth', 2, 'DisplayName', results(i).name);
    end
    legend('Location', 'northwest');
    xlabel('Time (days)'); ylabel('Total Cells');
    title('Decoupled Model Fits: Independent Saturation Control');
    axis([0 15 0 max(N_data)*1.5]);
end

%% --- Helper Functions ---

function r = model_residual(theta, model_func, params, t_data, N_data, tspan, tau, N0, model_name)
    N_model = simulate_model(theta, model_func, params, t_data, tspan, tau, N0, model_name);
    r = log(N_model(:)) - log(N_data(:));
end

function N_model = simulate_model(theta, model_func, params, t_eval, tspan, tau, N0, model_name)
    % 1. Extract Initial Fractions (Common to all)
    f_csc = theta(1);
    f_pc  = theta(2);
    f_tdc = max(0, 1 - f_csc - f_pc);
    
    % 2. Decoupled Parameter Mapping
    switch model_name
        case 'Combined'
            params.gamma01 = 10^theta(3);
            params.gamma11 = params.gamma01 * 10;
            params.beta0   = 10^theta(4);
            params.beta1   = params.beta0 * 0.05;
        case 'TypeII'
            params.gamma01 = 10^theta(3);
            params.gamma11 = params.gamma01 * 10;
            % Ensure beta has no effect
            params.beta0 = 0; params.beta1 = 0;
        case 'TypeI'
            params.beta0 = 10^theta(3);
            params.beta1 = params.beta0 * 0.1;
            % Ensure gamma has no effect
            params.gamma01 = 0; params.gamma11 = 0;
        case 'Basic'
            params.gamma01 = 0; params.beta0 = 0;
    end
    
    % 3. DDE Simulation
    history = N0 * [f_csc; f_pc; f_tdc];
    opts = ddeset('RelTol', 1e-5, 'AbsTol', 1e-8);
    sol = dde23(@(t,y,Z) model_func(t,y,Z,params), tau, history, tspan, opts);
    y = deval(sol, t_eval);
    N_model = sum(y, 1);
end

%% --- DDE Definitions ---
function dydt = dde_combined(t,y,Z,p)
    x0 = y(1); x1 = y(2); x2 = y(3); x2_delayed = Z(3,1);
    p0_eff = p.p0 / (1 + p.gamma01 * x2_delayed^2);
    q0_eff = p.q0 / (1 + p.gamma02 * x2_delayed^2);
    p1_eff = p.p1 / (1 + p.gamma11 * x2_delayed^2);
    q1_eff = p.q1 / (1 + p.gamma12 * x2_delayed^2);
    v0_eff = p.v0 / (1 + p.beta0 * x2_delayed^2);
    v1_eff = p.v1 / (1 + p.beta1 * x2_delayed^2);
    dydt = [(p0_eff - q0_eff)*v0_eff*x0 - p.d0*x0;
            (1 - p0_eff + q0_eff)*v0_eff*x0 + (p1_eff - q1_eff)*v1_eff*x1 - p.d1*x1;
            (1 - p1_eff + q1_eff)*v1_eff*x1 - p.d2*x2];
end

function dydt = dde_typeII(t,y,Z,p)
    x0 = y(1); x1 = y(2); x2 = y(3); x2_delayed = Z(3,1);
    p0_eff = p.p0 / (1 + p.gamma01 * x2_delayed^2);
    q0_eff = p.q0 / (1 + p.gamma02 * x2_delayed^2);
    p1_eff = p.p1 / (1 + p.gamma11 * x2_delayed^2);
    q1_eff = p.q1 / (1 + p.gamma12 * x2_delayed^2);
    dydt = [(p0_eff - q0_eff)*p.v0*x0 - p.d0*x0;
            (1 - p0_eff + q0_eff)*p.v0*x0 + (p1_eff - q1_eff)*p.v1*x1 - p.d1*x1;
            (1 - p1_eff + q1_eff)*p.v1*x1 - p.d2*x2];
end

function dydt = dde_typeI(t,y,Z,p)
    x0 = y(1); x1 = y(2); x2 = y(3); x2_delayed = Z(3,1);
    v0_eff = p.v0 / (1 + p.beta0 * x2_delayed^2);
    v1_eff = p.v1 / (1 + p.beta1 * x2_delayed^2);
    dydt = [(p.p0 - p.q0)*v0_eff*x0 - p.d0*x0;
            (1 - p.p0 + p.q0)*v0_eff*x0 + (p.p1 - p.q1)*v1_eff*x1 - p.d1*x1;
            (1 - p.p1 + p.q1)*v1_eff*x1 - p.d2*x2];
end

function dydt = dde_basic(~,y,~,p)
    x0 = y(1); x1 = y(2); x2 = y(3);
    dydt = [(p.p0 - p.q0)*p.v0*x0 - p.d0*x0;
            (1 - p.p0 + p.q0)*p.v0*x0 + (p.p1 - p.q1)*p.v1*x1 - p.d1*x1;
            (1 - p.p1 + p.q1)*p.v1*x1 - p.d2*x2];
end

% ====================

% Load data from files
M10 = readmatrix('data/tumor_growth_10nM.csv');
time_days = M10(:, 1);
data_10nM = M10(:, 2:end);

M20 = readmatrix('data/tumor_growth_20nM.csv');
data_20nM = M20(:, 2:end);

for i=1:6
    results = fit_breast_cancer_models(time_days, data_10nM(:, i));
end

% ====================