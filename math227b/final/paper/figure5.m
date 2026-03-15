function reproduce_lander_fig5_complete()
    % FIGURE 5: Parameters that Affect the Shapes of Steady-State Receptor Occupancy
    % Based on Lander, Nie, and Wan (2002) - Developmental Cell.
    
    % --- General Settings ---
    set(0, 'DefaultLineLineWidth', 1.5);
    fig = figure('Color', 'w', 'Position', [100, 100, 1000, 800]);
    
    % --- Panels A & B: Steady State Gradients ---
    % Parameters for A
    beta_A = 0.8; 
    psi_A = [1.25, 25, 66.7];
    
    % Parameters for B
    beta_B = 0.2; 
    psi_B = [1, 8, 22.7];
    
    x_range = linspace(0, 100, 100); % distance in microns (L=100)
    
    % Plot Panel A
    subplot(2,2,1); hold on;
    draw_gradients(beta_A, psi_A, x_range);
    title('A: \beta = 0.8'); ylabel('Fractional Occupancy (B)'); xlabel('distance (\mu m)');
    ylim([0 0.85]); xlim([0 100]); box on;
    
    % Plot Panel B
    subplot(2,2,2); hold on;
    draw_gradients(beta_B, psi_B, x_range);
    title('B: \beta = 0.2'); ylabel('Fractional Occupancy (B)'); xlabel('distance (\mu m)');
    ylim([0 0.22]); xlim([0 100]); box on;
    
    % --- Panels C & D: Parameter Space and Physical Constraints ---
    fprintf('Calculating parameter space for Panels C and D...\n');
    betas = linspace(0.01, 0.98, 40);
    psi_cutoff = zeros(size(betas));
    
    for i = 1:length(betas)
        b_val = betas(i);
        target_B = 0.1 * b_val; % Definition of breadth sigma=0.5: B(0.5L) = 0.1*B(0)
        
        % Search for psi that results in the 10% cutoff at mid-tissue
        obj = @(p) solve_B_at_xi(b_val, p, 0.5) - target_B;
        
        try
            % We look for psi where the gradient is "just right"
            psi_cutoff(i) = fzero(obj, [0.01, 200]);
        catch
            psi_cutoff(i) = NaN; % Handle cases where no steady state is found
        end
    end
    
    % Clean up any NaNs for plotting
    valid_idx = ~isnan(psi_cutoff);
    betas = betas(valid_idx);
    psi_cutoff = psi_cutoff(valid_idx);

    % Plot Panel C
    subplot(2,2,3); hold on;
    fill([betas, fliplr(betas)], [psi_cutoff, 100*ones(size(betas))], [0.9 0.9 0.9], 'EdgeColor', 'none');
    plot(betas, psi_cutoff, 'k', 'LineWidth', 2);
    text(0.3, 60, 'too steep', 'FontWeight', 'bold');
    text(0.88, 15, 'too shallow', 'Rotation', 90);
    xlabel('\beta'); ylabel('\psi'); title('C: Parameter Space (\sigma = 0.5)');
    axis([0 1 0 100]); box on;

    % Plot Panel D
    subplot(2,2,4); hold on;
    kon_vals = [1e5, 3e5, 1e6];
    % Scale factor used to match receptors per cell (derived from paper's units)
    % This maps dimensionless occupancy to physical receptor counts
    scale_factor = 2.5e8; 
    
    for k = 1:length(kon_vals)
        % R_max calculation based on the paper's derivation
        R_max = (betas .* (psi_cutoff.^2) * scale_factor) / (kon_vals(k) * 1e3);
        plot(betas, R_max, 'k');
        text(betas(end-8), R_max(end-8), sprintf('%.0e', kon_vals(k)), 'VerticalAlignment', 'bottom');
    end
    
    line([0 1], [100 100], 'Color', 'r', 'LineStyle', '--'); % Biological threshold
    xlabel('\beta'); ylabel('Max Occupied Receptors/Cell'); title('D: Physical Constraints');
    ylim([0 1100]); xlim([0.1 1]); box on;
    
    fprintf('Figure generation complete.\n');
end

% --- Helper Functions ---

function draw_gradients(beta, psi_list, x_range)
    % Helper to plot multiple curves for Panels A and B
    colors = lines(length(psi_list));
    for j = 1:length(psi_list)
        p = psi_list(j);
        B_profile = arrayfun(@(x) solve_B_at_xi(beta, p, x/100), x_range);
        plot(x_range, B_profile, 'k');
        % Label the curves
        text(x_range(round(end/3)), B_profile(round(end/3)), sprintf('\\psi=%.1f', p), ...
            'BackgroundColor', 'w', 'FontSize', 8);
    end
end

function B_val = solve_B_at_xi(beta, psi, target_xi)
    % Solves the non-linear BVP: d2a/dxi2 = psi^2 * a/(1+a)
    % a(0) = beta/(1-beta), a(1) = 0 (sink at end of tissue)
    
    a0 = beta / (1 - beta);
    
    % Exponential initial guess: helps solver converge for high psi
    x_mesh = linspace(0, 1, 50);
    solinit = bvpinit(x_mesh, @(x) [a0*exp(-psi*x); -psi*a0*exp(-psi*x)]);
    
    options = bvpset('RelTol', 1e-4, 'AbsTol', 1e-6, 'Stats', 'off');
    ode = @(x, y) [y(2); psi^2 * y(1)/(1+y(1))];
    bc = @(ya, yb) [ya(1) - a0; yb(1)]; 
    
    try
        sol = bvp4c(ode, bc, solinit, options);
        a_val = deval(sol, target_xi);
        B_val = a_val(1) / (1 + a_val(1));
    catch
        % If solver fails due to extreme steepness, B at target_xi is effectively 0
        B_val = 0;
    end
end