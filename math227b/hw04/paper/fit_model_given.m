function results = fit_breast_cancer_models(t_data, N_data)

    %% =============================================================
    %% Settings
    %% =============================================================
    
    tau = 1.2;
    tspan = [min(t_data) max(t_data)];
    
    N0 = N_data(1);
    
    %% =============================================================
    %% Initial guess for parameters to fit
    %% =============================================================
    
    % Only fitting initial fractions for now
    % [f_csc, f_pc]
    
    theta0 = [0.05 0.1];
    
    lb = [0 0];
    ub = [1 1];
    
    % options = optimoptions('lsqnonlin','Display','iter','MaxIterations',1000);
    options = optimoptions('lsqnonlin', ...
        'Display','iter', ...
        'MaxIterations',1000, ...
        'MaxFunctionEvaluations',5000, ...
        'FunctionTolerance',1e-10, ...
        'StepTolerance',1e-10, ...
        'OptimalityTolerance',1e-10);
    
    %% =============================================================
    %% Define parameter sets
    %% =============================================================
    
    params_combined = struct(...
    'p0',0.5,'q0',0.2,...
    'p1',0.5,'q1',0.1,...
    'v0',1.0,'v1',2.0,...
    'd0',0.01,'d1',0.05,'d2',0.1,...
    'gamma01',1e-14,'gamma02',1e-16,...
    'gamma11',1e-13,'gamma12',1e-15,...
    'beta0',8e-12,'beta1',4e-13);
    
    params_typeII = struct(...
    'p0',0.5,'q0',0.2,...
    'p1',0.5,'q1',0.1,...
    'v0',1.0,'v1',2.0,...
    'd0',0.01,'d1',0.05,'d2',0.1,...
    'gamma01',2e-14,'gamma02',2e-15,...
    'gamma11',4e-13,'gamma12',4e-15);
    
    params_typeI = struct(...
    'p0',0.5,'q0',0.2,...
    'p1',0.5,'q1',0.1,...
    'v0',1.0,'v1',2.0,...
    'd0',0.01,'d1',0.05,'d2',0.1,...
    'beta0',2e-11,'beta1',3e-12);
    
    params_basic = struct(...
    'p0',0.25,'q0',0.2,...
    'p1',0.3,'q1',0.1,...
    'v0',1.0,'v1',2.0,...
    'd0',0.01,'d1',0.05,'d2',0.1);
    
    %% =============================================================
    %% Model list
    %% =============================================================
    
    models = { ...
    struct('name','Combined','func',@dde_combined,'params',params_combined,'color','r'), ...
    struct('name','TypeII','func',@dde_typeII,'params',params_typeII,'color','g'), ...
    struct('name','TypeI','func',@dde_typeI,'params',params_typeI,'color','b'), ...
    struct('name','Basic','func',@dde_basic,'params',params_basic,'color','k')};
    
    %% =============================================================
    %% Fit each model
    %% =============================================================
    
    results = struct;
    
    for i = 1:length(models)
        m = models{i};
        objfun = @(theta) model_residual(theta,m.func,m.params,t_data,N_data,tspan,tau,N0);
        theta_fit = lsqnonlin(objfun,theta0,lb,ub,options);
    
        results(i).theta = theta_fit;
        results(i).name = m.name;
        results(i).func = m.func;
        results(i).params = m.params;
        results(i).color = m.color;
    end
    
    %% =============================================================
    %% Plot results
    %% =============================================================
    
    figure; hold on; grid on;
    scatter(t_data,N_data,60,'k','filled')
    % tplot = linspace(min(t_data),max(t_data),400);
    
    tspan_plot = [0 1200];
    tplot = linspace(0, 1200, 2e3);
    for i = 1:length(results)
        theta = results(i).theta;
        N_model = simulate_model(theta,results(i).func,results(i).params,tplot,tspan_plot,tau,N0);
        plot(tplot,N_model,'Color',results(i).color,'LineWidth',2)
    end
    
    legend({'Data','Combined','TypeII','TypeI','Basic'},'Location','northwest');
    axis([0 20 0 4e6]);
    
    xlabel('Time')
    ylabel('Total Cells')

end


% ====================

function r = model_residual(theta,model_func,params,t_data,N_data,tspan,tau,N0)
    N_model = simulate_model(theta,model_func,params,t_data,tspan,tau,N0);
    r = log(N_model) - log(N_data);
end

% ====================

function N_model = simulate_model(theta,model_func,params,t_eval,tspan,tau,N0)

    f_csc = theta(1);
    f_pc  = theta(2);
    
    f_tdc = max(0,1 - f_csc - f_pc);
    
    history = N0*[f_csc;f_pc;f_tdc];
    
    sol = dde23(@(t,y,Z) model_func(t,y,Z,params),tau,history,tspan);
    
    y = deval(sol,t_eval);
    
    N_model = sum(y,1);

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

% ====================

observations = [ ...
    0.0,  0.20;
    1.0,  0.10;
    2.0,  0.14;
    3.0,  0.23;
    4.0,  0.27;
    5.0,  0.28;
    6.0,  0.47;
    7.0,  0.70;
    8.0,  1.12;
    9.0,  1.57;
   10.0,  2.15;
   12.0,  3.00;
   13.0,  3.05;
   14.0,  3.02];
t_data = observations(:, 1); 
N_data = observations(:, 2)*1e6;
results = fit_breast_cancer_models(t_data,N_data);

% ====================