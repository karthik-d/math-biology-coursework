function breast_cancer_fig1_typeII
    % =============================================================
    % 1. PARAMETERS FOR TYPE II FEEDBACK
    % =============================================================
    p.p0 = 0.5; p.q0 = 0.2;    % Basal CSC division probabilities
    p.p1 = 0.5; p.q1 = 0.1;    % Basal PC division probabilities
    p.v0 = 1.0; p.v1 = 2.0;    % Constant proliferation rates
    p.d0 = 0.01; p.d1 = 0.05; p.d2 = 0.1; % Death rates
    
    % Feedback strengths for Type II (Division Probabilities)
    % Based on Table S2: gamma values are used for p and q
    p.gamma01 = 1e-14;  % Delayed feedback strength on CSC division
    p.gamma02 = 1e-16;
    p.gamma11 = 1e-13;
    p.gamma12 = 1e-15;

    p.beta0 = 8e-12;
    p.beta1 = 4e-13;
    tau = 2;           % Time delay (days)

    % =============================================================
    % 2. SOLVE THE DDE SYSTEM
    % =============================================================
    tspan = [0 20];
    history = [1.8e2; 0; 0]; % Initial condition: 10,000 CSCs
    
    % dde23(DDE_FUNC, DELAYS, HISTORY, TSPAN)
    sol = dde23(@(t, y, Z) dde_typeII_logic(t, y, Z, p), tau, history, tspan);

    % =============================================================
    % 3. PLOTTING
    % =============================================================
    total_cells = sum(sol.y, 1);
    % csc_percent = (sol.y(1,:) ./ total_cells) * 100;
    
    figure('Color', 'w');
    
    % Subplot 1: Growth Kinetics (matching Figure 1 style)
    subplot(1,1,1);
    plot(sol.x, total_cells, 'g-', 'LineWidth', 2);
    grid on;
    ylabel('Total Cell Number');
    title('Figure 1: Type II Feedback Growth Kinetics');
    axis([0 20 0 4e6]);
    
    % Subplot 2: CSC Percentage (matching Figure 2 style)
    % subplot(2,1,2);
    % plot(sol.x, csc_percent, 'r-', 'LineWidth', 2);
    % grid on;
    % ylabel('CSC %');
    % xlabel('Culture Time (days)');
    % title('Figure 2: CSC Population Dynamics (Type II)');
end

% =================================================================
% TYPE II DDE SYSTEM LOGIC
% =================================================================
function dydt = dde_typeII_logic(t, y, Z, p)
    % Current states: x0=y(1), x1=y(2), x2=y(3)
    % Delayed state: x2(t - tau) is Z(3,1)

    x0 = y(1);
    x1 = y(2);
    x2 = y(3);
    
    x2_delayed = Z(3,1);
    
    % Type II Feedback: TDC inhibits symmetric division probabilities
    % Eq (S3) in Supplementary: p_eff = p / (1 + gamma * x2^2)
    p0_eff = p.p0 / (1 + p.gamma01 * x2_delayed^2);
    q0_eff = p.q0 / (1 + p.gamma02 * x2_delayed^2);
    
    p1_eff = p.p1 / (1 + p.gamma11 * x2_delayed^2);
    q1_eff = p.q1 / (1 + p.gamma12 * x2_delayed^2);
    
    % Synthesis rates remain constant (Unlike Type I)
    fb1 = 1 / (1 + p.beta0 * (x2_delayed^2));
    fb2 = 1 / (1 + p.beta1 * (x2_delayed^2));
    
    % Effective proliferation rates
    v0_eff = p.v0 * fb1;
    v1_eff = p.v1 * fb2;
    
    % DDE System
    dx0dt = (p0_eff - q0_eff) * v0_eff * x0 - p.d0 * x0;
    
    dx1dt = (1 - p0_eff + q0_eff) * v0_eff * x0 + ...
            (p1_eff - q1_eff) * v1_eff * x1 - p.d1 * x1;
            
    dx2dt = (1 - p1_eff + q1_eff) * v1_eff * x1 - p.d2 * x2;
    
    dydt = [dx0dt; dx1dt; dx2dt];
end