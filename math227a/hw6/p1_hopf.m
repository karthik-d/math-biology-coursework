% System: x' = y + mu*x;  y' = -x + mu*y - x^2*y

% 1. Simulation at the critical point, mu = 0
mu = 0;
f = @(t,X) [ X(2) + mu*X(1);  -X(1) + mu*X(2) - X(1).^2 .* X(2) ];
% direction field arrows
[xg,yg] = meshgrid(-2:0.3:2, -2:0.3:2);
u = yg + mu*xg;
v = -xg + mu*yg - xg.^2 .* yg;

figure; hold on; title('Phase portrait at \mu = 0');
xlabel('x'); ylabel('y'); axis equal; grid on;
% Draw arrows scaled for clarity
quiver(xg, yg, u, v, 1.5, 'Color', [0.3 0.3 0.3]);

tspan = [0 500];
X0s = [0.5 0.1; 1.0 0.2; 0.2 0.8];
for k=1:size(X0s,1)
    [t,X] = ode45(f, tspan, X0s(k,:)');
    plot(X(:,1), X(:,2), 'LineWidth', 1.2);
end
legend('Vector field','IC1','IC2','IC3');
prettyfig;


% 2. Simulation for a small positive mu to show the limit cycle.
mu = 0.05;    % small positive value for which a stable limit cycle exists
f = @(t,X) [ X(2) + mu*X(1);  -X(1) + mu*X(2) - X(1).^2 .* X(2) ];
% Vector field
[xg,yg] = meshgrid(-2:0.3:2, -2:0.3:2);
u = yg + mu*xg;
v = -xg + mu*yg - xg.^2 .* yg;

% Simulation
tspan = [0 1000];
X0 = [1.5; 0.0];
[t,X] = ode45(f, tspan, X0);
% Separate trajectory into transient and limit cycle portion
transient = round(length(t)*0.6);

figure; hold on;
quiver(xg, yg, u, v, 1.5, 'Color', [0.3 0.3 0.3]);   % vector field arrows
plot(X(:,1), X(:,2), 'b-', 'LineWidth', 1);          % full trajectory
plot(X(transient:end,1), X(transient:end,2), 'r-', 'LineWidth', 2);  % limit cycle emphasized

title(['Full trajectory and limit cycle for \mu = ', num2str(mu)]);
xlabel('x'); ylabel('y'); axis equal; grid on;
legend('Vector field','Full trajectory','Limit cycle');
prettyfig;
