function results = fit_breast_cancer_proportions_final()
    %% 1. Data Setup (From image_4932dd.jpg)
    observations = [
         0.0, 1.5; 2.0, 2.1; 4.0, 4.2; 6.0, 3.9; 
         8.0, 2.2; 10.0, 1.3; 12.0, 0.9
    ];
    t_data = observations(:, 1);
    prop_data = observations(:, 2) / 100; % Convert % to fraction

    %% 2. Optimization Settings
    % theta = [f_csc, f_pc, log10(gamma), log10(beta)]
    % Note: We ensure lb and ub are at least 0.2 apart where possible.
    theta0 = [0.02, 0.10, -10, -10]; 
    lb     = [0.00, 0.00, -20, -20]; % Lower bounds
    ub     = [0.30, 0.50,  -1,  -1]; % Upper bounds (Ensuring > 0.2 span)
    
    options = optimoptions('lsqnonlin', ...
        'Display', 'iter', ...
        'FunctionTolerance', 1e-10, ...
        'StepTolerance', 1e-10, ...
        'OptimalityTolerance', 1e-10, ...
        'FiniteDifferenceStepSize', 0.1, ... % Explicitly set step size
        'TypicalX', [0.02, 0.1, -10, -10]);   % Help solver scale steps

    %% 3. Model Definitions
    base_p = struct('p0',0.5,'q0',0.2,'p1',0.5,'q1',0.1,'v0',1.0,'v1',2.0,...
                    'd0',0.01,'d1',0.05,'d2',0.1,'h',2,'tau',1.2);

    models = { ...
        struct('name','Combined','func',@dde_combined,'color','r'), ...
        struct('name','Type II','func',@dde_typeII,'color','g') ...
        struct('name','Type I','func',@dde_typeI,'color','b') ...
        struct('name','Basic','func',@dde_basic,'color','k') ...
    };

    %% 4. Perform Fitting
    results = struct;
    for i = 1:length(models)
        m = models{i};
        fprintf('\n--- Fitting Model: %s ---\n', m.name);
        
        objfun = @(theta) model_residual(theta, m.func, base_p, t_data, prop_data);
        [theta_fit, ~] = lsqnonlin(objfun, theta0, lb, ub, options);
    
        results(i).theta = theta_fit;
        results(i).name = m.name;
        results(i).func = m.func;
        results(i).color = m.color;
    end

    %% 5. Visualization
    figure('Color','w'); hold on; grid on;
    scatter(t_data, prop_data*100, 80, 'r', 'filled', 'DisplayName', 'Observation');
    
    t_plot = linspace(0, 15, 200);
    for i = 1:length(results)
        p_fit = update_params(base_p, results(i).theta);
        prop_model = simulate_model(results(i).theta, results(i).func, p_fit, t_plot);
        plot(t_plot, prop_model*100, 'Color', results(i).color, 'LineWidth', 2, 'DisplayName', results(i).name);
    end
    xlabel('Time (days)'); ylabel('CSC %'); legend('Location','best');
end

%% --- Helper Functions ---

function r = model_residual(theta, model_func, base_p, t_data, prop_data)
    p_curr = update_params(base_p, theta);
    prop_model = simulate_model(theta, model_func, p_curr, t_data);
    r = prop_model - prop_data;
end

function p = update_params(base_p, theta)
    p = base_p;
    p.gamma01 = 10^theta(3); p.gamma02 = 10^theta(3);
    p.gamma11 = 10^theta(3); p.gamma12 = 10^theta(3);
    p.beta0 = 10^theta(4);   p.beta1 = 10^theta(4);
end

function prop_csc = simulate_model(theta, model_func, p, t_eval)
    N0 = 1e6;
    history = N0 * [theta(1); theta(2); max(0, 1 - theta(1) - theta(2))];
    try
        opts = ddeset('RelTol', 1e-5, 'AbsTol', 1e-8);
        sol = dde23(@(t,y,Z) model_func(t,y,Z,p), p.tau, history, [0, max(t_eval)], opts);
        y = deval(sol, t_eval);
        prop_csc = (y(1,:) ./ sum(y, 1))';
    catch
        prop_csc = zeros(size(t_eval)); 
    end
end

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