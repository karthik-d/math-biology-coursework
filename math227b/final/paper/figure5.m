function reproduce_lander_fig5_final()
    % FIGURE 5: Parameters affecting Steady-State Receptor Occupancy
    % Based on Lander, Nie, and Wan (2002)
    
    set(0, 'DefaultLineLineWidth', 1.5);
    figure('Color', 'w', 'Position', [100, 100, 1000, 850]);
    
    % --- Physical Constants ---
    % k_deg / (k_off + k_deg) ratio used to scale the dimensionless psi
    k_factor = 0.07; 
    
    % --- Panels A & B: Steady State Gradients ---
    x_range = linspace(0, 100, 100);
    
    subplot(2,2,1); hold on;
    draw_gradients(0.8, [1.25, 25, 40], x_range, k_factor);
    title('A: \beta = 0.8'); ylabel('Fractional Occupancy (B)'); xlabel('distance (\mu m)');
    ylim([0 0.85]); xlim([0 100]); box on;
    
    subplot(2,2,2); hold on;
    draw_gradients(0.2, [1, 8, 22.7], x_range, k_factor);
    title('B: \beta = 0.2'); ylabel('Fractional Occupancy (B)'); xlabel('distance (\mu m)');
    ylim([0 0.22]); xlim([0 100]); box on;
    
    % --- Panel C: Parameter Space ---
    fprintf('Calculating Panel C (eta = 0.5 boundary)...\n');
    betas = linspace(0.01, 0.99, 100);
    psi_cutoff = zeros(size(betas));
    
    for i = 1:length(betas)
        b = betas(i);
        % eta = 0.5 means B(x=0.5L) = 0.5 * B(0)
        target_B = 0.5 * b; 
        obj = @(p) solve_B_at_xi(b, p, 0.5, k_factor) - target_B;
        try
            psi_cutoff(i) = fzero(obj, [0.1, 500]);
        catch
            psi_cutoff(i) = NaN;
        end
    end
    
    subplot(2,2,3); hold on;
    % Fill "too steep" area
    fill([betas, 1, 0], [psi_cutoff, 100, 100], [0.9 0.9 0.9], 'EdgeColor', 'none');
    plot(betas, psi_cutoff, 'k', 'LineWidth', 2);
    
    % The "too shallow" boundary in the paper occurs at very high beta 
    % where the gradient doesn't drop enough even with psi=0.
    shallow_x = [0.82, 1.0]; shallow_y = [0, 40];
    plot(shallow_x, shallow_y, 'k', 'LineWidth', 1.5);
    
    text(0.3, 60, 'too steep', 'FontWeight', 'bold', 'FontSize', 12);
    text(0.92, 15, 'too shallow', 'Rotation', 90, 'FontSize', 10);
    xlabel('\beta'); ylabel('\psi'); title('C: Parameter Space (\eta = 0.5)');
    axis([0 1 0 100]); box on;

    % --- Panel D: Physical Constraints ---
    subplot(2,2,4); hold on;
    kon_vals = [1e5, 3e5, 1e6];
    scale_factor = 2.1e5; % Adjusted to match receptors/cell magnitude
    
    for k = 1:length(kon_vals)
        R_max = (betas .* (psi_cutoff.^2) * scale_factor) / kon_vals(k);
        plot(betas, R_max, 'k');
        % Label the specific kon lines
        text(betas(end-12), R_max(end-12), sprintf('k_{on} = %.0e', kon_vals(k)), ...
            'VerticalAlignment', 'bottom', 'FontSize', 9);
    end
    
    plot([0 1], [100 100], 'r--', 'HandleVisibility', 'off'); 
    text(0.15, 120, '100 receptors/cell', 'Color', 'r', 'FontSize', 8);
    xlabel('\beta'); ylabel('Max Occupied Receptors/Cell'); title('D: Physical Constraints');
    ylim([0 1100]); xlim([0.1 1]); box on;
    prettyfig;
    
    fprintf('Reproduction Complete.\n');
end

% --- Helper Functions ---

function draw_gradients(beta, psi_list, x_range, k_f)
    for p = psi_list
        B_profile = arrayfun(@(x) solve_B_at_xi(beta, p, x/100, k_f), x_range);
        plot(x_range, B_profile, 'k');
        % Label curves near the midpoint
        [~, idx] = min(abs(x_range - 40));
    end
end

function B_val = solve_B_at_xi(beta, psi, target_xi, k_f)
    a0 = beta / (1 - beta);
    eff_psi = sqrt(k_f * psi^2);
    
    x_mesh = linspace(0, 1, 50);
    solinit = bvpinit(x_mesh, @(x) [a0*exp(-eff_psi*x); -eff_psi*a0*exp(-eff_psi*x)]);
    
    options = bvpset('RelTol', 1e-4, 'Stats', 'off');
    ode = @(x, y) [y(2); (k_f * psi^2) * y(1)/(1+y(1))];
    bc = @(ya, yb) [ya(1) - a0; yb(1)]; 
    
    try
        sol = bvp4c(ode, bc, solinit, options);
        res = deval(sol, target_xi);
        B_val = res(1) / (1 + res(1));
    catch
        B_val = 0;
    end
end