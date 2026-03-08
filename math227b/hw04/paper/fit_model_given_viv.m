%% --- DRIVER SCRIPT ---
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

group_names = {'H605 #1', 'H605 #2', 'MCF7/HER2 #1', 'MCF7/HER2 #2'};

for i = 2:5
    col = data_matrix(:, i);
    valid = ~isnan(col);
    if sum(valid) > 2
        t_data = data_matrix(valid, 1);
        N_data = col(valid) .* 1e11; 
        fprintf('\n--- Fitting %s ---\n', group_names{i-1});
        fit_breast_cancer_models(t_data(:), N_data(:), group_names{i-1});
    end
end

%% --- MAIN FITTING FUNCTION ---
function results = fit_breast_cancer_models(t_data, N_data, title_str)
    tau = 1.2;
    tspan = [0 max(t_data)];
    N0 = max(N_data(1), 1e7); % Minimum starting population
    
    % Initial guesses for [f_csc, f_pc]
    theta0 = [0.1, 0.2];
    lb = [1e-5, 1e-5];
    ub = [0.9, 0.9];
    
    % CRITICAL CHANGE: DiffMinChange forced to 1e-2 to jump over flat spots
    options = optimoptions('lsqnonlin', ...
        'Display','iter', ...
        'MaxIterations', 200, ...
        'StepTolerance', 1e-12, ...
        'DiffMinChange', 1e-2, ... 
        'FunctionTolerance', 1e-10);

    % Adjusted Proliferation Rates: Increased v0/v1 to match rapid late-stage growth
    params_combined = struct('p0',0.6,'q0',0.1,'p1',0.5,'q1',0.1,'v0',1.8,'v1',2.5,...
                             'd0',0.01,'d1',0.05,'d2',0.1,...
                             'gamma01',1e-24,'gamma02',1e-24,...
                             'gamma11',1e-24,'gamma12',1e-24,...
                             'beta0',1e-13,'beta1',1e-14);
    
    params_basic = struct('p0',0.6,'q0',0.1,'p1',0.5,'q1',0.1,'v0',1.8,'v1',2.5,...
                          'd0',0.01,'d1',0.05,'d2',0.1);

    models = { ...
        struct('name','Feedback','func',@dde_combined,'params',params_combined,'color','r'), ...
        struct('name','No-Feedback','func',@dde_basic,'params',params_basic,'color','k')};
    
    figure; hold on; grid on;
    scatter(t_data, N_data, 70, 'k', 'filled', 'DisplayName', 'Data');

    for i = 1:length(models)
        m = models{i};
        % Rescaling the residual internally to keep the solver in a 'healthy' range
        objfun = @(theta) (log(simulate_model(theta,m.func,m.params,t_data,tspan,tau,N0)+1e6) - log(N_data+1e6));
        
        try
            theta_fit = lsqnonlin(objfun,theta0,lb,ub,options);
            tplot = linspace(0, max(t_data), 150);
            N_model = simulate_model(theta_fit, m.func, m.params, tplot, tspan, tau, N0);
            plot(tplot, N_model, 'Color', m.color, 'LineWidth', 2.5, 'DisplayName', m.name);
        catch ME
            fprintf('Model %s failed: %s\n', m.name, ME.message);
        end
    end
    
    legend('Location', 'northwest');
    xlabel('Time (days)'); ylabel('Total Cells');
    title(['Optimization Results: ', title_str]);
    axis([0 120 0 7e11]);
end

%% --- DDE FUNCTIONS ---
function dydt = dde_combined(t,y,Z,p)
    x0 = y(1); x1 = y(2); x2 = y(3);
    x2_delayed = Z(3,1);
    % Feedback Logic
    p0_eff = p.p0 / (1 + p.gamma01 * x2_delayed^2);
    q0_eff = p.q0 / (1 + p.gamma02 * x2_delayed^2);
    v0_eff = p.v0 / (1 + p.beta0 * x2_delayed);
    v1_eff = p.v1 / (1 + p.beta1 * x2_delayed);
    dydt = [(p0_eff - q0_eff)*v0_eff*x0 - p.d0*x0;
            (1 - p0_eff + q0_eff)*v0_eff*x0 + (p.p1 - p.q1)*v1_eff*x1 - p.d1*x1;
            (1 - p.p1 + p.q1)*v1_eff*x1 - p.d2*x2];
end

function dydt = dde_basic(t,y,~,p)
    dydt = [(p.p0 - p.q0)*p.v0*y(1) - p.d0*y(1);
            (1 - p.p0 + p.q0)*p.v0*y(1) + (p.p1 - p.q1)*p.v1*y(2) - p.d1*y(2);
            (1 - p.p1 + p.q1)*p.v1*y(2) - p.d2*y(3)];
end

%% --- SIMULATION CORE ---
function N_model = simulate_model(theta,model_func,params,t_eval,tspan,tau,N0)
    history = N0 * [theta(1); theta(2); max(0, 1-theta(1)-theta(2))];
    opts = ddeset('RelTol',1e-4,'AbsTol',1e-7);
    sol = dde23(@(t,y,Z) model_func(t,y,Z,params), tau, history, tspan, opts);
    y_out = deval(sol, t_eval);
    N_model = sum(y_out, 1);
end