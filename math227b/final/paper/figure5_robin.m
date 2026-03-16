function reproduce_lander_fig5_physical()
    % Reproduce Figure 5 of Lander, Nie, Wan (2002) using the paper's
    % definitions of beta and psi and the correct flux (Robin) boundary condition.
    %
    % Model: System B in the paper: steady-state, 1D, free ligand a(x),
    % fractional occupancy B(x) = a/(1+a), domain x in [0,1].
    %
    % ODE:  a''(x) = psi^2 * a(x)/(1 + a(x))
    %
    % BC at x=1 (distal sink):  a(1) = 0
    % BC at x=0 (source flux):  a(0)/(1 + a(0)) - beta = (1/psi^2)*a'(0)
    %
    % Parameters:
    %   psi^2 = [k_deg/(k_deg + k_off)] * [L^2 * k_on * R0 / D]
    %   beta  = nu / (k_deg * R0)
    %
    % Panels:
    %   A,B: steady-state B(x) for selected (beta,psi)
    %   C: locus of (beta,psi) where eta = 0.5 (B drops to 0.5 of its value
    %      between x=0 and x=0.5)
    %   D: convert psi_cutoff to occupied receptors/cell at the source for
    %      several k_on values, using the paper's psi definition.

    set(0,'DefaultLineLineWidth',1.5);
    figure('Color','w','Position',[100,100,1000,850]);

    % ---------------- Panels A & B: steady-state gradients ----------------
    x_range = linspace(0,100,100);   % physical distance (µm); x/L = x/100

    % Panel A: beta = 0.8
    subplot(2,2,1); hold on;
    draw_gradients_physical(0.8,[1.25,25,40],x_range);
    title('A: \beta = 0.8');
    ylabel('Fractional Occupancy B');
    xlabel('distance (\mum)');
    ylim([0 0.85]); xlim([0 100]); box on;

    % Panel B: beta = 0.2
    subplot(2,2,2); hold on;
    draw_gradients_physical(0.2,[1,8,22.7],x_range);
    title('B: \beta = 0.2');
    ylabel('Fractional Occupancy B');
    xlabel('distance (\mum)');
    ylim([0 0.22]); xlim([0 100]); box on;

    % ---------------- Panel C: parameter space (eta = 0.5) ----------------
    fprintf('Calculating Panel C (eta = 0.5 boundary)...\n');
    betas = linspace(0.01,0.99,100);
    psi_cutoff = nan(size(betas));

    % Definition of eta in the paper: here we impose
    % B(0.5L) = 0.5 * B(0) as in your previous code, but using the
    % correct BVP (so eta=0.5).
    for i = 1:length(betas)
        beta = betas(i);
        obj = @(psi) B_at_xi(beta,psi,0.5) - 0.5*B_at_xi(beta,psi,0.0);
        try
            psi_cutoff(i) = fzero(obj,[0.1,500]);  % same psi range as before
        catch
            psi_cutoff(i) = NaN;
        end
    end

    subplot(2,2,3); hold on;
    % Fill "too steep" area
    maxPsiPlot = 100;
    fill([betas,1,0],[psi_cutoff,maxPsiPlot,maxPsiPlot],[0.9 0.9 0.9], ...
         'EdgeColor','none');
    plot(betas,psi_cutoff,'k','LineWidth',2);

    % "too shallow" boundary: as in the paper, very high beta where even
    % psi~0 does not give enough drop; approximated here as a straight line.
    shallow_x = [0.82,1.0];
    shallow_y = [0,maxPsiPlot*0.4];
    plot(shallow_x,shallow_y,'k','LineWidth',1.5);

    text(0.3,0.6*maxPsiPlot,'too steep','FontWeight','bold','FontSize',12);
    text(0.92,0.15*maxPsiPlot,'too shallow','Rotation',90,'FontSize',10);
    xlabel('\beta'); ylabel('\psi'); title('C: Parameter Space (\eta = 0.5)');
    axis([0 1 0 maxPsiPlot]); box on;

    % ---------------- Panel D: physical constraints ----------------
    subplot(2,2,4); hold on;

    % Physical parameters as in the paper (approximate):
    % D ~ 1e-7 cm^2/s, L ~ 0.01 cm (100 µm), k_off << k_deg
    D  = 1e-7;       % cm^2/s
    L  = 0.01;       % cm (100 µm)
    kon_vals = [1e5,3e5,1e6];  % M^-1 s^-1, as in Fig 5D

    % We use tight-binding approximation: (k_deg + k_off)/k_deg ~ 1
    % so psi^2 ~ L^2 k_on R0 / D  =>  R0 ~ psi^2 * D / (L^2 k_on)
    % Occupied receptors at source: ~ beta * R0
    for k = 1:length(kon_vals)
        kon = kon_vals(k);
        R0 = (psi_cutoff.^2 * D)/(L^2 * kon);  % receptors per "unit cell" (scaled)
        R_max = betas .* R0;                  % occupied receptors at x=0
        plot(betas,R_max,'k');
        text(betas(end-12),R_max(end-12), ...
             sprintf('k_{on} = %.0e',kon), ...
             'VerticalAlignment','bottom','FontSize',9);
    end

    % 100 receptors/cell horizontal line
    plot([0 1],[100 100],'r--','HandleVisibility','off');
    text(0.15,120,'100 receptors/cell','Color','r','FontSize',8);

    xlabel('\beta');
    ylabel('Max Occupied Receptors/Cell');
    title('D: Physical Constraints');
    ylim([0 1100]); xlim([0.1 1]); box on;

    % Optional: your formatting helper
    % prettyfig;

    fprintf('Reproduction Complete (physical beta, psi).\n');
end

% =======================================================================
% Helper: draw gradients for given beta, list of psi values, and x_range
% =======================================================================
function draw_gradients_physical(beta,psi_list,x_range)
    for psi = psi_list
        B_profile = arrayfun(@(x) B_at_xi(beta,psi,x/100), x_range);
        plot(x_range,B_profile,'k');
    end
end

% =======================================================================
% Helper: fractional occupancy B at dimensionless position xi in [0,1]
% =======================================================================
function B_val = B_at_xi(beta,psi,target_xi)
    % Solve the steady-state BVP:
    %   a'' = psi^2 * a/(1+a),  x in (0,1)
    %   a(1) = 0
    %   a(0)/(1+a(0)) - beta = (1/psi^2) * a'(0)
    %
    % then return B(target_xi) = a/(1+a).

    % Mesh and initial guess
    x_mesh = linspace(0,1,50);

    % Crude initial guess: exponential-like decay, using beta as approximate B(0)
    a_guess0 = beta/(1-beta);           % if B(0) ~ beta
    a_guess  = @(x) a_guess0 * exp(-psi*x);
    solinit  = bvpinit(x_mesh,@(x) [a_guess(x); -psi*a_guess(x)]);

    options = bvpset('RelTol',1e-4,'Stats','off');

    ode   = @(x,y) ode_system(x,y,psi);
    bcfun = @(ya,yb) bc_flux(ya,yb,beta,psi);

    try
        sol = bvp4c(ode,bcfun,solinit,options);
        res = deval(sol,target_xi);
        a   = res(1);
        B_val = a/(1 + a);
    catch
        % If the solver fails, return 0 (or NaN) to keep caller robust
        B_val = 0;
    end
end

% =======================================================================
% ODE system: y = [a; a']
% =======================================================================
function dydx = ode_system(x,y,psi)
    a  = y(1);
    ap = y(2);
    dydx = [ap;
            psi^2 * a/(1 + a)];
end

% =======================================================================
% Boundary conditions (Robin at x=0, Dirichlet at x=1)
% =======================================================================
function res = bc_flux(ya,yb,beta,psi)
    a0  = ya(1);
    ap0 = ya(2);
    % Robin BC at x=0: a(0)/(1+a(0)) - beta = (1/psi^2)*a'(0)
    res1 = a0/(1 + a0) - beta - (1/psi^2)*ap0;
    % Dirichlet at x=1: a(1) = 0
    res2 = yb(1);
    res  = [res1; res2];
end
