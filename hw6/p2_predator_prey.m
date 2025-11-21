% Parameters
b = 3; % example value, must be >2 for Hopf
a_c = 4*(b-2)/(b^2*(b+2)); % Hopf bifurcation value
fprintf('Hopf bifurcation occurs at a_c = %.4f\n', a_c);

% System
predprey = @(t,Y,a) [b*Y(1) - Y(1)^2 - (Y(1)*Y(2))/(1+Y(1));
                     -a*Y(2)^2 + (Y(1)*Y(2))/(1+Y(1))];

%% Compute positive fixed point (y* in terms of x*)
x_fixed_fun = @(x,a) b - x - (x./(a*(1+x)^2));
x_guess = 0.5;
xstar = fzero(@(x) x_fixed_fun(x,a_c), x_guess);
ystar = xstar/(a_c*(1+xstar));
fprintf('Positive fixed point at a_c: x* = %.4f, y* = %.4f\n', xstar, ystar);

%% Time integration
tspan = [0 200];

% Below Hopf
a1 = a_c*0.9;
Y0 = [xstar*0.8; ystar*0.8]; % initial condition
[t1,Y1] = ode45(@(t,Y) predprey(t,Y,a1), tspan, Y0);

% Above Hopf
a2 = a_c*1.1;
Y0 = [xstar*1.2; ystar*1.2]; % initial condition
[t2,Y2] = ode45(@(t,Y) predprey(t,Y,a2), tspan, Y0);

%% Figure 1: below Hopf
figure;
plot(Y1(:,1), Y1(:,2), 'b', 'LineWidth', 1.5)
hold on
plot(Y1(1,1), Y1(1,2), 'go', 'MarkerFaceColor','g', 'MarkerSize',8) % initial
plot(xstar, ystar, 'ro', 'MarkerFaceColor','r', 'MarkerSize',8)      % fixed point
plot(Y1(end,1), Y1(end,2), 'ko', 'MarkerFaceColor','k', 'MarkerSize',8) % final
xlabel('x (prey)')
ylabel('y (predator)')
title(sprintf('Phase portrait: a = %.3f < a_c', a1))
legend('Trajectory','Initial','Steady state','Final','Location','best')
grid on
prettyfig;

%% Figure 2: above Hopf
figure;
plot(Y2(:,1), Y2(:,2), 'r', 'LineWidth', 1.5)
hold on
plot(Y2(1,1), Y2(1,2), 'go', 'MarkerFaceColor','g', 'MarkerSize',8) % initial
plot(xstar, ystar, 'ro', 'MarkerFaceColor','r', 'MarkerSize',8)      % fixed point
plot(Y2(end,1), Y2(end,2), 'ko', 'MarkerFaceColor','k', 'MarkerSize',8) % final
xlabel('x (prey)')
ylabel('y (predator)')
title(sprintf('Phase portrait: a = %.3f > a_c', a2))
legend('Trajectory','Initial','Steady state','Final','Location','best')
grid on
prettyfig;