function verify_solver()
% Generic 1D reaction-diffusion test suite for solver verification.
% This version is self-contained: all helper functions are defined below.

systems = define_test_systems();
for k = 1:numel(systems)
    sys = systems{k};
    [x, t, U] = solve_rd_system(sys);
    plot_test_result(sys, x, t, U, k);
end
end

function systems = define_test_systems()
systems = {
    fisher_kpp_system(), ...
    schnakenberg_system(), ...
    three_field_cascade_system() ...
    };
end

function sys = fisher_kpp_system()
sys.name = 'Scalar Fisher-KPP benchmark';
sys.nvar = 1;
sys.L = 60;
sys.nx = 220;
sys.tspan = [0 20];
sys.tplot = [0 4 8 12 16 20];
sys.D = 0.5;
sys.params.r = 1.0;
sys.icfun = @(x) 1./(1 + exp((x - 10)/1.5));
sys.reaction = @(x,U,p) p.r*U(1,:).*(1 - U(1,:));
end

function sys = schnakenberg_system()
sys.name = 'Schnakenberg front / pattern benchmark';
sys.nvar = 2;
sys.L = 40;
sys.nx = 240;
sys.tspan = [0 5];
sys.tplot = [0 1 2 3 4 5];
sys.D = [1.0, 0.01];
sys.params.a = 0.1;
sys.params.b = 0.9;
u_ss = sys.params.b/(sys.params.a + sys.params.b)^2;
v_ss = sys.params.a + sys.params.b;
sys.icfun = @(x) [u_ss*ones(size(x)); v_ss + 0.6*exp(-((x - sys.L/2)/1.5).^2)];
sys.reaction = @(x,U,p) [p.b - U(1,:).*U(2,:).^2; p.a + U(1,:).*U(2,:).^2 - U(2,:)];
end

function sys = three_field_cascade_system()
sys.name = 'Three-field morphogen cascade';
sys.nvar = 3;
sys.L = 40;
sys.nx = 240;
sys.tspan = [0 80];
sys.tplot = [0 10 20 40 80];
sys.D = [0.35, 0.18, 0.08];
sys.params.mu = [0.12, 0.08, 0.05];
sys.params.k12 = 0.22;
sys.params.k23 = 0.16;
sys.params.source = @(x) 1.8*exp(-(x/2.5).^2);
sys.icfun = @(x) zeros(3, numel(x));
sys.reaction = @(x,U,p) [ ...
    p.source(x) - (p.k12 + p.mu(1))*U(1,:); ...
    p.k12*U(1,:) - (p.k23 + p.mu(2))*U(2,:); ...
    p.k23*U(2,:) - p.mu(3)*U(3,:) ];
end

function [x, tplot, Uout] = solve_rd_system(sys)
x = linspace(0, sys.L, sys.nx);
dx = x(2) - x(1);
L = neumann_laplacian(sys.nx, dx);
A = blkdiag_sparse(sys.D, L);
U0 = sys.icfun(x);
y0 = U0(:);
rhs = @(t,y) rhs_rd(t, y, sys, x, A);
opts = odeset('RelTol',1e-6,'AbsTol',1e-8);
sol = ode15s(rhs, sys.tspan, y0, opts);
tplot = sys.tplot;
Y = deval(sol, tplot);
Uout = reshape(Y, sys.nvar, sys.nx, []);
Uout = permute(Uout, [3 1 2]);
end

function dydt = rhs_rd(~, y, sys, x, A)
U = reshape(y, sys.nvar, sys.nx);
R = sys.reaction(x, U, sys.params);
dydt = A*y + R(:);
end

function L = neumann_laplacian(nx, dx)
e = ones(nx,1);
L = spdiags([e -2*e e], -1:1, nx, nx);
L(1,1) = -2;  L(1,2) = 2;
L(end,end) = -2; L(end,end-1) = 2;
L = L/(dx^2);
end

function A = blkdiag_sparse(D, L)
D = D(:);
blocks = cell(numel(D),1);
for j = 1:numel(D)
    blocks{j} = D(j)*L;
end
A = blkdiag(blocks{:});
end

function plot_test_result(sys, x, tplot, Uout, fig_id)
figure(fig_id); clf; hold on;
cols = parula(numel(tplot));
if sys.nvar == 3
    plot(x, squeeze(Uout(end,1,:)), 'LineWidth', 2);
    plot(x, squeeze(Uout(end,2,:)), 'LineWidth', 2);
    plot(x, squeeze(Uout(end,3,:)), 'LineWidth', 2);
    legend({'u','v','w'}, 'Location', 'northeast');
else
    field_to_plot = min(sys.nvar, 2);
    for k = 1:numel(tplot)
        plot(x, squeeze(Uout(k,field_to_plot,:)), 'Color', cols(k,:), 'LineWidth', 1.8);
    end
    legend(compose('t = %.2g', tplot), 'Location', 'best');
end
xlabel('x'); ylabel('Concentration');
title(sys.name);
set(gca,'Box','off','TickDir','out');
end
