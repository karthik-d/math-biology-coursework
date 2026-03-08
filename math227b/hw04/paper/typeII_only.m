% Clear workspace
clear; clc;

% --- Parameters ---
p.v0 = 0.7895;
p.v1 = 0.4566;
p.d0 = 0.1788;
p.d1 = 0.2000;
p.d2 = 0.0500;
p.p0 = 0.9500;
p.q0 = 0.0500;
p.p1 = 0.8500;
p.q1 = 0.0500;
p.gamma01 = 0.2240;
p.gamma11 = 0.1500;
p.gamma02 = 0;
p.gamma12 = 0;

tau = 1.0; % Time delay in days
tspan = [0 1020];

% --- Initial Conditions (History) ---
% History must be a function or a constant vector for t <= 0
% Initial population: 0.08 units of x0, 0.02 of x1 (Total 0.1)
history = [0.08; 0.02; 0.0]; 

% --- Solve DDE ---
sol = dde23(@(t,y,Z) dde_typeII(t,y,Z,p), tau, history, tspan);

% --- Plotting ---
figure('Color', 'w');
total_cells = sum(sol.y, 1); % Sum of x0 + x1 + x2

plot(sol.x, total_cells, 'y-', 'LineWidth', 2.5);
hold on;
grid on;

% Labeling
xlabel('Time (days)');
ylabel('Cell Count (10^6 cells)');
title('Type II Feedback DDE Fit (Yellow Curve)');
legend('Total Population (x0+x1+x2)');


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