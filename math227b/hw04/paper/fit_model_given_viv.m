function fit_breast_cancer_dde_final()
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
    group_names = {'H605 #1', 'H605 #2', 'MCF7/HER2 #1', 'MCF7/HER2 #2'};

    for i = 2:5
        col = data_matrix(:, i);
        valid = ~isnan(col);
        if sum(valid) > 2
            t_data = data_matrix(valid, 1);
            t_start = t_data(1);
            t_data_norm = t_data - t_start;
            N_data = (col(valid) .* 1e11) + 1e5; 
            
            fprintf('\n--- Fitting %s ---\n', group_names{i-1});
            perform_fit_with_proportions(t_data_norm(:), N_data(:), group_names{i-1}, t_start);
        end
    end
end

function perform_fit_with_proportions(t_data, N_data, name, t_offset)
    % p_log = [beta0, beta1, gamma01, gamma02, gamma11, gamma12, v0, v1, init_total]
    p_guess_log = [-25, -25, -22, -22, -22, -22, log10(0.8), log10(1.5), log10(N_data(1))];
    lb = [-40, -40, -40, -40, -40, -40, log10(0.1), log10(0.1), 5]; 
    ub = [-10, -10, -10, -10, -10, -10, log10(5.0), log10(5.0), 14];

    options = optimoptions('lsqcurvefit', 'Display', 'iter', 'DiffMinChange', 0.1, ...
        'FunctionTolerance', 1e-10, 'OptimalityTolerance', 1e-10);

    % Perform the fit using the log-residual helper
    [p_fit_log, ~] = lsqcurvefit(@(p, t) model_helper(p, t), p_guess_log, t_data, log10(N_data), lb, ub, options);

    %% Final Simulation for Plotting
    t_fine = linspace(0, max(t_data), 200)';
    [N_fit, CSC_fit, PC_fit, TDC_fit] = get_full_model_state(p_fit_log, t_fine);
    CSC_percent = 100 * (CSC_fit ./ N_fit);

    %% Figure 1: Linear Total Cell Fit
    figure('Color', 'w', 'Name', [name ' - Total Cells']);
    plot(t_data + t_offset, N_data, 'ko', 'MarkerFaceColor', 'k', 'DisplayName', 'Data'); hold on;
    plot(t_fine + t_offset, N_fit, 'r-', 'LineWidth', 2, 'DisplayName', 'DDE Fit');
    xlabel('Time (days)'); ylabel('Total Cells');
    title(['Total Cell Number: ' name]); grid on; legend('Location', 'best');
    xlim([50 120]);
    prettyfig;

    %% Figure 2: CSC Proportion
    figure('Color', 'w', 'Name', [name ' - CSC Proportion']);
    plot(t_fine + t_offset, CSC_percent, 'b-', 'LineWidth', 2);
    xlabel('Time (days)'); ylabel('CSC (%)');
    title(['CSC Proportion Over Time: ' name]);
    grid on; ylim([0 100]); xlim([50 120]);
    prettyfig;

    %% Helper: Returns only log10(Total) for lsqcurvefit
    function log_N = model_helper(p_log, t_vector)
        [N, ~, ~, ~] = get_full_model_state(p_log, t_vector);
        log_N = log10(N + 1);
    end

    %% Main Solver Engine
    function [Total, CSC, PC, TDC] = get_full_model_state(p_log, t_vector)
        p_val = 10.^p_log;
        ps.beta0 = p_val(1); ps.beta1 = p_val(2);
        ps.gamma01 = p_val(3); ps.gamma02 = p_val(4);
        ps.gamma11 = p_val(5); ps.gamma12 = p_val(6);
        ps.v0 = p_val(7); ps.v1 = p_val(8);
        init_val = p_val(9);

        ps.p0 = 0.5; ps.q0 = 0.2; ps.p1 = 0.5; ps.q1 = 0.1;
        ps.d0 = 0.01; ps.d1 = 0.05; ps.d2 = 0.10;
        ps.h = 2; tau = 2;
        
        % History [CSC; PC; TDC]
        history = init_val * [0.70; 0.20; 0.10];

        try
            opts = ddeset('RelTol', 1e-5, 'AbsTol', 1e-8);
            sol = dde23(@(t,y,Z) dde_rhs(t,y,Z,ps), tau, history, [0, max(t_vector)], opts);
            y_interp = deval(sol, t_vector);
            CSC = y_interp(1, :)';
            PC  = y_interp(2, :)';
            TDC = y_interp(3, :)';
            Total = CSC + PC + TDC;
        catch
            Total = zeros(length(t_vector), 1) + 1e-5;
            CSC = Total; PC = Total; TDC = Total;
        end
    end
end

function dydt = dde_rhs(t, y, Z, p)
    x0 = y(1); x1 = y(2); x2 = y(3);
    x2_del = Z(3,1);
    
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