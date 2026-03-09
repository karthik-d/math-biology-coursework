function fit_csc_proportion_dde()
    %% 1. Data Setup (CSC % CD44+CD24-)
    % Column 1: Time (days), Column 2: CSC %
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

    fprintf('\n--- Fitting CSC Proportion Dataset ---\n');
    perform_csc_fit(t_data, y_data);
end

function perform_csc_fit(t_data, y_data)
    % Parameters: [beta0, beta1, gamma01, gamma02, gamma11, gamma12, v0, v1, init_total_log]
    % We use log10 scales for rates to keep search space stable
    p_guess_log = [-1, -1, -2, -2, -2, -2, log10(0.8), log10(1.5), 5]; 
    lb = [-5, -5, -6, -6, -6, -6, log10(0.01), log10(0.01), 2]; 
    ub = [ 2,  2,  2,  2,  2,  2, log10(5.0), log10(5.0), 10];

    options = optimoptions('lsqcurvefit', 'Display', 'iter', 'UseParallel', false, ...
        'FunctionTolerance', 1e-8, 'OptimalityTolerance', 1e-8);

    % Fit the model
    [p_fit_log, ~] = lsqcurvefit(@(p, t) model_helper(p, t), p_guess_log, t_data, y_data, lb, ub, options);

    %% Final Simulation for Plotting
    t_fine = linspace(0, max(t_data)+1, 200)';
    [~, CSC_fit, PC_fit, TDC_fit] = get_full_model_state(p_fit_log, t_fine);
    
    % Calculate Percentage for plotting
    Total_fit = CSC_fit + PC_fit + TDC_fit;
    CSC_percent_fit = 100 * (CSC_fit ./ Total_fit);

    %% Visualization
    figure('Color', 'w', 'Name', 'CSC Proportion Fit');
    plot(t_data, y_data, 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Observation'); hold on;
    plot(t_fine, CSC_percent_fit, 'b-', 'LineWidth', 2, 'DisplayName', 'DDE Simulation');
    xlabel('Culture time (days)'); ylabel('CD44+CD24- (%)');
    title('Fit to CSC Proportion'); 
    grid on; ylim([0 100]); xlim([0 15]); legend('Location', 'best');

    %% Helper: Returns CSC % for lsqcurvefit
    function csc_pct = model_helper(p_log, t_vector)
        [Total, CSC, ~, ~] = get_full_model_state(p_log, t_vector);
        csc_pct = 100 * (CSC ./ Total);
        % Handle cases where solver fails
        csc_pct(isnan(csc_pct)) = 0;
    end

    %% Main Solver Engine
    function [Total, CSC, PC, TDC] = get_full_model_state(p_log, t_vector)
        p_val = 10.^p_log;
        ps.beta0 = p_val(1); ps.beta1 = p_val(2);
        ps.gamma01 = p_val(3); ps.gamma02 = p_val(4);
        ps.gamma11 = p_val(5); ps.gamma12 = p_val(6);
        ps.v0 = p_val(7); ps.v1 = p_val(8);
        init_total = p_val(9);
        
        % Fixed biological parameters
        ps.p0 = 0.5; ps.q0 = 0.2; ps.p1 = 0.5; ps.q1 = 0.1;
        ps.d0 = 0.01; ps.d1 = 0.05; ps.d2 = 0.10;
        ps.h = 2; tau = 2;
        
        % History [CSC; PC; TDC] - Start with ~79% CSC to match day 0
        history = init_total * [0.79; 0.11; 0.10];
        
        try
            opts = ddeset('RelTol', 1e-4, 'AbsTol', 1e-6);
            sol = dde23(@(t,y,Z) dde_rhs(t,y,Z,ps), tau, history, [0, max(t_vector)], opts);
            y_interp = deval(sol, t_vector);
            CSC = y_interp(1, :)';
            PC  = y_interp(2, :)';
            TDC = y_interp(3, :)';
            Total = CSC + PC + TDC;
        catch
            Total = ones(length(t_vector), 1);
            CSC = zeros(length(t_vector), 1); 
            PC = Total; TDC = Total;
        end
    end
end

function dydt = dde_rhs(t, y, Z, p)
    x0 = y(1); x1 = y(2); x2 = y(3);
    x2_del = Z(3,1); % Delayed TDC concentration
    
    % Feedback mechanisms
    p0_e = p.p0 / (1 + p.gamma01 * x2_del^p.h);
    q0_e = p.q0 / (1 + p.gamma02 * x2_del^p.h);
    p1_e = p.p1 / (1 + p.gamma11 * x2_del^p.h);
    q1_e = p.q1 / (1 + p.gamma12 * x2_del^p.h);
    v0_e = p.v0 / (1 + p.beta0 * x2_del^p.h);
    v1_e = p.v1 / (1 + p.beta1 * x2_del^p.h);

    dx0 = (p0_e - q0_e) * v0_e * x0 - p.d0 * x0;
    dx1 = (1 - p0_e + q0_e) * v0_e * x0 + (p1_e - q1_e) * v1_e * x1 - p.d1 * x1;
    dx2 = (1 - p1_e + q1_e) * v1_e * x1 - p.d2 * x2;
    dydt = [dx0; dx1; dx2];
end