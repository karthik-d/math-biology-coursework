function breast_cancer_fig1_combined
    % =============================================================
    % 1. ALL FB.
    % =============================================================
    tau = 2;           % Time delay (days)
    tspan = [0 200];
    history = [50000; 0; 0]; % Initial condition: 10,000 CSCs
    sol = dde23(@(t, y, Z) dde_combined_logic(t, y, Z), tau, history, tspan);

    total_cells = sum(sol.y, 1);
    figure('Color', 'w'); hold on;
    plot(sol.x, total_cells, 'r-', 'LineWidth', 2);
    plot(sol.x, sol.y);
    legend('0', '1', '2', '3');
    axis([0 30 0 4e7]);
    ylabel('Total Cell Number');
    grid on;
    title('Figure 1: Type II Feedback Growth Kinetics');

    % =============================================================
    % 2. TYPE I
    % =============================================================
    tau = 2;               % Time delay (days)
    tspan = [0 200];
    history = [1.8e2; 0; 0]; % Initial condition: 10,000 CSCs
    sol = dde23(@(t, y, Z) dde_logic(t, y, Z), tau, history, tspan);

    total_cells = sum(sol.y, 1);
    % plot(sol.x, total_cells, 'b-', 'LineWidth', 2);

    % =============================================================
    % 3. TYPE II
    % =============================================================
    tau = 2;           % Time delay (days)
    tspan = [0 200];
    history = [1.8e2; 0; 0]; % Initial condition: 10,000 CSCs
    sol = dde23(@(t, y, Z) dde_typeII_logic(t, y, Z), tau, history, tspan);

    total_cells = sum(sol.y, 1);
    % plot(sol.x, total_cells, 'g-', 'LineWidth', 2);

    % =============================================================
    % 4. BASIC
    % =============================================================          % Time delay (days)
    tspan = [0 200];
    history = [1.8e2; 0; 0]; % Initial condition: 10,000 CSCs
    sol = dde23(@(t, y, Z) dde_basic_logic(t, y, Z), tau, history, tspan);

    total_cells = sum(sol.y, 1);
    % plot(sol.x, total_cells, 'k--', 'LineWidth', 2);
    % legend("Combined", "Type I", "Type II", "Basic");
end

% =================================================================
% TYPE II DDE SYSTEM LOGIC
% =================================================================
function dydt = dde_combined_logic(t, y, Z)
    % --- Fixed Parameters (Based on Table S2 and Fig 1c/2b) ---
    p.p0 = 0.5; p.q0 = 0.2;    
    p.p1 = 0.5; p.q1 = 0.1;    
    p.v0 = 1.0; p.v1 = 2.0;    
    p.d0 = 0.01; p.d1 = 0.05; p.d2 = 0.1; 
    
    % MISTAKE CORRECTION: gamma01 must equal gamma02 for monotonic absolute growth.
    % Using values consistent with the MCF7 model best-fits.
    p.gamma01 = 1e-14;  
    p.gamma02 = 1e-16; 
    p.gamma11 = 1e-13;
    p.gamma12 = 1e-15;
    % p.gamma01 = 1e-23;  
    % p.gamma02 = 2e-24; 
    % p.gamma11 = 4e-22;
    % p.gamma12 = 5e-23;
    % p.gamma02 = p.gamma01;
    % p.gamma12 = p.gamma11;
    
    % Beta parameters for synthesis inhibition (Type I)
    p.beta0 = 8e-12;
    p.beta1 = 4e-13;
    
    % p.beta0 = p.beta1;
    % p.beta0 = 8e-27;
    % p.beta1 = 4e-27;

    % Current and Delayed States
    x0 = y(1);
    x1 = y(2);
    x2 = y(3);
    x2_delayed = Z(3,1); % TDC at t - tau
    % x2_delayed = y(3);
    
    % --- Type II Feedback: Symmetric Division Probabilities ---
    % Correction: p0 and q0 should be inhibited by the same factor
    % to prevent the growth rate (p-q) from crossing below zero.
    p0_eff = p.p0 / (1 + p.gamma01 * (x2_delayed^2));
    q0_eff = p.q0 / (1 + p.gamma02 * (x2_delayed^2));
    
    p1_eff = p.p1 / (1 + p.gamma11 * (x2_delayed^2));
    q1_eff = p.q1 / (1 + p.gamma12 * (x2_delayed^2));
    
    % --- Type I Feedback: Proliferation/Synthesis Rates ---
    fb1 = 1 / (1 + p.beta0 * (x2_delayed^2));
    fb2 = 1 / (1 + p.beta1 * (x2_delayed^2));
    
    v0_eff = p.v0 * fb1;
    v1_eff = p.v1 * fb2;
    
    % --- System of Equations (Equation S4) ---
    % CSCs
    dx0dt = (p0_eff - q0_eff) * v0_eff * x0 - p.d0 * x0;
    dx1dt = (1 - p0_eff + q0_eff) * v0_eff * x0 + ...
            (p1_eff - q1_eff) * v1_eff * x1 - p.d1 * x1;
    dx2dt = (1 - p1_eff + q1_eff) * v1_eff * x1 - p.d2 * x2;
    
    dydt = [dx0dt; dx1dt; dx2dt];
end


function dydt = dde_typeII_logic(t, y, Z)
    p.p0 = 0.5; p.q0 = 0.2;    % Basal CSC division probabilities
    p.p1 = 0.5; p.q1 = 0.1;    % Basal PC division probabilities
    p.v0 = 1.0; p.v1 = 2.0;    % Constant proliferation rates
    p.d0 = 0.01; p.d1 = 0.05; p.d2 = 0.1; % Death rates
    %
    p.gamma01 = 5e-14;  % Delayed feedback strength on CSC division
    p.gamma02 = 7e-15;
    p.gamma11 = 6e-13;
    p.gamma12 = 2e-15;
    
    % p.gamma02 = p.gamma01;
    % p.gamma12 = p.gamma11;
    % Current states: x0=y(1), x1=y(2), x2=y(3)
    % Delayed state: x2(t - tau) is Z(3,1)
    x0 = y(1);
    x1 = y(2);
    x2 = y(3);
    x2_delayed = Z(3,1);
    
    p0_eff = p.p0 / (1 + p.gamma01 * x2_delayed^2);
    q0_eff = p.q0 / (1 + p.gamma02 * x2_delayed^2);
    
    p1_eff = p.p1 / (1 + p.gamma11 * x2_delayed^2);
    q1_eff = p.q1 / (1 + p.gamma12 * x2_delayed^2);
    
    % Synthesis rates remain constant (Unlike Type I)
    v0 = p.v0;
    v1 = p.v1;
    
    % DDE System
    dx0dt = (p0_eff - q0_eff) * v0 * x0 - p.d0 * x0;
    dx1dt = (1 - p0_eff + q0_eff) * v0 * x0 + ...
            (p1_eff - q1_eff) * v1 * x1 - p.d1 * x1;
    dx2dt = (1 - p1_eff + q1_eff) * v1 * x1 - p.d2 * x2;
    dydt = [dx0dt; dx1dt; dx2dt];
end


function dydt = dde_logic(t, y, Z)
    p.p0 = 0.5; p.q0 = 0.2;    % CSC division probabilities
    p.p1 = 0.5; p.q1 = 0.1;    % PC division probabilities
    p.v0 = 1.0; p.v1 = 2.0;    % Proliferation rates
    p.d0 = 0.01; p.d1 = 0.05; p.d2 = 0.1; % Death rates
    %
    p.beta0 = 2e-11;  % Delayed feedback strength (from user spec)
    p.beta1 = 3e-12;  % Instantaneous feedback strength (from user spec)
    
    % p.beta0 = p.beta1;

    x0 = y(1);
    x1 = y(2);
    x2 = y(3);
    x2_delayed = Z(3, 1); % TDC population at t - tau
    
    % Feedback terms based on your provided corrected model:
    % fb1 uses delayed TDC (x2_delayed)
    % fb2 uses instantaneous TDC (x2)
    fb1 = 1 / (1 + p.beta0 * (x2_delayed^2));
    fb2 = 1 / (1 + p.beta1 * (x2_delayed^2));
    
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


function dydt = dde_basic_logic(t, y, Z)
    p.p0 = 0.25; p.q0 = 0.2;    % CSC division probabilities
    p.p1 = 0.3; p.q1 = 0.1;    % PC division probabilities
    p.v0 = 1.0; p.v1 = 2.0;    % Proliferation rates
    p.d0 = 0.01; p.d1 = 0.05; p.d2 = 0.1; % Death rates
    
    x0 = y(1);
    x1 = y(2);
    x2 = y(3);
    
    % Effective proliferation rates
    v0_eff = p.v0;
    v1_eff = p.v1;
    
    % Equations (S2 Variation)
    dx0dt = (p.p0 - p.q0) * v0_eff * x0 - p.d0 * x0;
    dx1dt = (1 - p.p0 + p.q0) * v0_eff * x0 + ...
            (p.p1 - p.q1) * v1_eff * x1 - p.d1 * x1;
    dx2dt = (1 - p.p1 + p.q1) * v1_eff * x1 - p.d2 * x2;
    dydt = [dx0dt; dx1dt; dx2dt];
end