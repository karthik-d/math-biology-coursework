function reproduce_lander_fig5_final()
    % FIGURE 5: Parameters affecting Steady-State Receptor Occupancy
    % Improved version with colored curves, colored regions, and legends

    figure('Color', 'w', 'Position', [100, 100, 1050, 850]);

    % --- Physical Constants ---
    k_factor = 0.07;

    % --- Plot settings ---
    x_range = linspace(0, 100, 250);
    psi_colors = [0.15 0.45 0.85;
                  0.90 0.55 0.15;
                  0.80 0.20 0.20];
    region_colors.tooSteep   = [0.95 0.78 0.78];
    region_colors.acceptable = [0.82 0.93 0.82];
    region_colors.tooShallow = [0.98 0.90 0.65];
    kon_colors = [0.10 0.40 0.75;
                  0.00 0.62 0.45;
                  0.80 0.25 0.20];

    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    % -------------------------
    % Panel A
    % -------------------------
    nexttile; hold on;
    psiA = [1.25, 25, 66.7];
    hA = draw_gradients(0.8, psiA, x_range, k_factor, psi_colors);
    title('\beta = 0.8', 'FontWeight', 'normal');
    xlabel('distance (\mum)');
    ylabel('Occupancy ([LR]/R_{tot})');
    xlim([0 100]); ylim([0 0.8]);
    legend(hA, psi_legend_strings(psiA), ...
        'Location', 'southwest', 'Box', 'off', 'FontSize', 10);
    format_axes(gca);
    add_panel_label(gca, 'A.');

    % -------------------------
    % Panel B
    % -------------------------
    nexttile; hold on;
    psiB = [1, 8, 22.7];
    hB = draw_gradients(0.2, psiB, x_range, k_factor, psi_colors);
    title('\beta = 0.2', 'FontWeight', 'normal');
    xlabel('distance (\mum)');
    ylabel('Occupancy ([LR]/R_{tot})');
    xlim([0 100]); ylim([0 0.2]);
    legend(hB, psi_legend_strings(psiB), ...
        'Location', 'southwest', 'Box', 'off', 'FontSize', 10);
    format_axes(gca);
    add_panel_label(gca, 'B.');

    % -------------------------
    % Panel C
    % -------------------------
    fprintf('Calculating Panel C (eta = 0.5 boundary)...\n');

    betas = linspace(0.01, 0.99, 180);
    psi_cutoff = nan(size(betas));

    for i = 1:length(betas)
        b = betas(i);
        target_B = 0.5 * b;   % eta = 0.5 => B(0.5L)=0.5*B(0)
        obj = @(p) solve_B_at_xi(b, p, 0.5, k_factor) - target_B;
        try
            psi_cutoff(i) = fzero(obj, [0.1, 500]);
        catch
            psi_cutoff(i) = NaN;
        end
    end

    nexttile; hold on;

    valid = ~isnan(psi_cutoff);
    beta_v = betas(valid);
    psi_v  = psi_cutoff(valid);

    % Too steep region: above cutoff curve
    hSteep = fill([beta_v, fliplr(beta_v)], ...
                  [psi_v, 100*ones(size(psi_v))], ...
                  region_colors.tooSteep, ...
                  'EdgeColor', 'none', 'FaceAlpha', 0.65, ...
                  'DisplayName', 'Too steep');

    % Acceptable region: below cutoff curve, excluding shallow wedge visually
    hGood = fill([beta_v, fliplr(beta_v)], ...
                 [zeros(size(psi_v)), fliplr(psi_v)], ...
                 region_colors.acceptable, ...
                 'EdgeColor', 'none', 'FaceAlpha', 0.55, ...
                 'DisplayName', 'Admissible');

    % Approximate "too shallow" wedge near beta ~ 1, based on reference figure
    beta_shallow = linspace(0.86, 1.0, 80);
    psi_shallow  = 45 * (beta_shallow - 0.86) / (1.0 - 0.86);
    hShallow = fill([beta_shallow, 1.0, 0.86], ...
                    [psi_shallow, 0, 0], ...
                    region_colors.tooShallow, ...
                    'EdgeColor', 'none', 'FaceAlpha', 0.80, ...
                    'DisplayName', 'Too shallow');

    % Boundary curves
    hCut = plot(beta_v, psi_v, 'Color', [0.20 0.20 0.20], ...
        'LineWidth', 2.0, 'DisplayName', '\eta = 0.5 cutoff');
    plot(beta_shallow, psi_shallow, '--', 'Color', [0.45 0.30 0.10], ...
        'LineWidth', 1.5, 'HandleVisibility', 'off');

    text(0.23, 62, 'too steep', 'FontSize', 11, 'FontWeight', 'bold', ...
        'Color', [0.45 0.10 0.10]);
    text(0.90, 13, 'too shallow', 'FontSize', 10, 'Rotation', 72, ...
        'Color', [0.55 0.35 0.05]);

    xlabel('\beta');
    ylabel('\psi');
    title('Parameter Space', 'FontWeight', 'normal');
    axis([0 1 0 100]);
    legend([hSteep, hGood, hShallow, hCut], ...
        {'Too steep', 'Admissible', 'Too shallow', '\eta = 0.5 cutoff'}, ...
        'Location', 'northwest', 'Box', 'off', 'FontSize', 10);
    format_axes(gca);
    add_panel_label(gca, 'C.');

    % -------------------------
    % Panel D
    % -------------------------
    nexttile; hold on;

    kon_vals = [1e5, 3e5, 1e6];
    scale_factor = 2.1e5;

    hD = gobjects(numel(kon_vals),1);
    for k = 1:length(kon_vals)
        R_max = (betas .* (psi_cutoff.^2) * scale_factor) / kon_vals(k);
        hD(k) = plot(betas, R_max, 'LineWidth', 2.2, ...
            'Color', kon_colors(k,:), ...
            'DisplayName', kon_label(kon_vals(k)));
    end

    hThresh = yline(100, '--', 'Color', [0.55 0.00 0.00], ...
        'LineWidth', 1.6, 'DisplayName', '100 receptors/cell');

    xlabel('\beta');
    ylabel('Max. Occupied Receptors/Cell');
    title('Physical Constraints', 'FontWeight', 'normal');
    xlim([0.1 1.0]);
    ylim([0 1150]);
    legend([hD(:); hThresh], 'Location', 'northwest', ...
        'Box', 'off', 'FontSize', 10);
    format_axes(gca);
    add_panel_label(gca, 'D.');
    prettyfig;

    fprintf('Reproduction complete.\n');
end

% =========================
% Helper functions
% =========================

function h = draw_gradients(beta, psi_list, x_range, k_f, line_colors)
    h = gobjects(numel(psi_list),1);
    for i = 1:numel(psi_list)
        p = psi_list(i);
        B_profile = arrayfun(@(x) solve_B_at_xi(beta, p, x/100, k_f), x_range);
        h(i) = plot(x_range, B_profile, ...
            'Color', line_colors(i,:), ...
            'LineWidth', 2.2);
    end
end

function labels = psi_legend_strings(psi_list)
    labels = cell(size(psi_list));
    for i = 1:numel(psi_list)
        labels{i} = sprintf('\\psi = %g', psi_list(i));
    end
end

function txt = kon_label(kon_val)
    if abs(kon_val - 1e5) < 1
        txt = 'k_{on} = 10^{5}';
    elseif abs(kon_val - 3e5) < 1
        txt = 'k_{on} = 3 \times 10^{5}';
    elseif abs(kon_val - 1e6) < 1
        txt = 'k_{on} = 10^{6}';
    else
        txt = sprintf('k_{on} = %.2g', kon_val);
    end
end

function format_axes(ax)
    set(ax, 'FontSize', 11, ...
            'LineWidth', 1.1, ...
            'Box', 'off', ...
            'TickDir', 'out', ...
            'Layer', 'top');
end

function add_panel_label(ax, labeltxt)
    text(ax, -0.28, 1.05, labeltxt, ...
        'Units', 'normalized', ...
        'FontSize', 28, ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'top');
end

function B_val = solve_B_at_xi(beta, psi, target_xi, k_f)
    a0 = beta / (1 - beta);
    eff_psi = sqrt(k_f * psi^2);

    x_mesh = linspace(0, 1, 60);
    solinit = bvpinit(x_mesh, @(x) [a0*exp(-eff_psi*x); -eff_psi*a0*exp(-eff_psi*x)]);

    options = bvpset('RelTol', 1e-4, 'AbsTol', 1e-7, 'Stats', 'off');
    ode = @(x, y) [y(2); (k_f * psi^2) * y(1)/(1+y(1))];
    bc  = @(ya, yb) [ya(1) - a0; yb(1)];

    try
        sol = bvp4c(ode, bc, solinit, options);
        res = deval(sol, target_xi);
        B_val = res(1) / (1 + res(1));
    catch
        B_val = NaN;
    end
end
