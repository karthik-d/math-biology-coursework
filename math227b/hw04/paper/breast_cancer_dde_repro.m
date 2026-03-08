function breast_cancer_dde_repro
    % =============================================================
    % 1. PARAMETERS FOR FIGURE 1C (MCF7)
    % =============================================================
    params.p0 = 0.5; params.q0 = 0.2;    % CSC division probabilities
    params.p1 = 0.5; params.q1 = 0.1;    % PC division probabilities
    params.v0 = 1.0; params.v1 = 2.0;    % Proliferation rates
    params.d0 = 0.01; params.d1 = 0.05; params.d2 = 0.1; % Death rates
    
    params.beta0 = 2e-11;  % Delayed feedback strength (from user spec)
    params.beta1 = 3e-12;  % Instantaneous feedback strength (from user spec)
    tau = 2;               % Time delay (days)

    % =============================================================
    % 2. SOLVE THE DDE
    % =============================================================
    tspan = [0 20];
    history = [1.8e2; 0; 0]; % 10,000 CSCs, 0 PCs, 0 TDCs
    
    % dde23(DDE_FUNC, DELAYS, HISTORY, TSPAN)
    sol = dde23(@(t, y, Z) dde_logic(t, y, Z, params), tau, history, tspan);

    % =============================================================
    % 3. PLOTTING
    % =============================================================
    total_cells = sum(sol.y, 1);
    
    figure;
    plot(sol.x, total_cells, 'b-', 'LineWidth', 2);
    hold on;
    grid on;
    
    title('Reproduction of Figure 1C: Type I Feedback Dynamics');
    xlabel('Culture Time (days)');
    ylabel('Total Cell Number (log scale)');
    axis([0 20 0 4e6]);
    legend('Type I (MATLAB dde23)');
end

% =================================================================
% DDE SYSTEM LOGIC
% =================================================================
function dydt = dde_logic(t, y, Z, p)
    % y is the current state [x0; x1; x2]
    % Z is the state at t - tau: Z(:,1) = [x0(t-tau); x1(t-tau); x2(t-tau)]
    
    x0 = y(1);
    x1 = y(2);
    x2 = y(3);
    
    x2_delayed = Z(3, 1); % TDC population at t - tau
    
    % Feedback terms based on your provided corrected model:
    % fb1 uses delayed TDC (x2_delayed)
    % fb2 uses instantaneous TDC (x2)
    fb1 = 1 / (1 + p.beta0 * (x2_delayed^2));
    fb2 = 1 / (1 + p.beta1 * (x2^2));
    
    
    % Effective proliferation rates
    v0_eff = p.v0 * fb1;
    v1_eff = p.v1 * fb2;
    
    % Equations (S2 Variation)
    dx0dt = (p.p0 - p.q0) * v0_eff * x0 - p.d0 * x0;
    
    dx1dt = (1 - p.p0 + p.q0) * v0_eff * x0 + ...
            (p.p1 - p.q1) * v1_eff * x1 - p.d1 * x1;
            
    dx2dt = (1 - p.p1 + p.q1) * v1_eff * x1 - p.d2 * x2;
    
    dydt = [dx0dt; dx1dt; dx2dt];
end