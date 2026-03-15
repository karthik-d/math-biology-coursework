function lander_reproduction_fig6_corrected()
    % --- Parameters ---
    D_val = 1e-7;            % Diffusion coefficient (cm^2/s)
    xmax_cm = 0.01;          % 100 microns = 0.01 cm
    
    % Rates (s^-1)
    konR0 = 0.012;           
    koff  = 1e-5;
    kdeg  = 3.3e-5;
    kp    = 6e-4;
    kq    = 5e-5;
    kin   = 6e-4;
    kout  = 6.7e-5;
    kg    = 1e-4;

    % CORRECTED Boundary Flux: 8e-7 cm/s (Fixes the paper's 8e-5 typo)
    % v_norm = 8e-7; 
    % v_norm = 2.66e-7;
    v_norm = 3e-7;

    % Initial Conditions for receptors (steady state empty receptors)
    D0 = 1.0;
    E0 = kp / kq; % Evaluates to 12

    % --- Spatial and Temporal Grids ---
    x = linspace(0, xmax_cm, 100);
    x_microns = x * 1e4; 
    
    % Simulation from 2 to 24 hours to capture the exactly 12 curves plotted in the paper
    t_hours = 2:2:24;
    t = t_hours * 3600; 

    % --- Solve using pdepe ---
    % u(1)=A (free), u(2)=B (surf-bound), u(3)=C (int-bound), 
    % u(4)=D (surf-recept), u(5)=E (int-recept)
    sol = pdepe(0, @pde_sys, @ic_sys, @bc_sys, x, t);

    A = sol(:,:,1);
    B = sol(:,:,2);
    C = sol(:,:,3);

    % --- Plotting ---
    figure('Color', 'w', 'Position', [100, 100, 900, 750]);

    % A: Free Morphogen
    subplot(2,2,1); plot_curves(x_microns, A, '[L]/R_0', 'free (A)', [0 0.01]);
    
    % B: Surface Bound
    subplot(2,2,2); plot_curves(x_microns, B, '[LR]_{out}/R_0', 'surface bound (B)', [0 0.4]);

    % C: Internal Bound
    subplot(2,2,3); plot_curves(x_microns, C, '[LR]_{in}/R_0', 'internal bound (C)', [0 2.5]);

    % D: Total Bound (B + C)
    subplot(2,2,4); plot_curves(x_microns, B+C, '([LR]_{out}+[LR]_{in})/R_0', 'total bound (B+C)', [0 3]);

    % Helper function for consistent plotting formatting
    function plot_curves(xv, yv, ylbl, titl, ylims)
        hold on;
        plot(xv, yv', 'k', 'LineWidth', 0.8);
        title(titl); xlabel('distance (microns)'); ylabel(ylbl);
        ylim(ylims); xlim([0 100]); grid on;
    end

    % --- PDE System Definition ---
    function [c, f, s] = pde_sys(~, ~, u, dudx)
        c = [1; 1; 1; 1; 1];
        f = [D_val * dudx(1); 0; 0; 0; 0]; 
        
        A=u(1); B=u(2); C=u(3); D=u(4); E=u(5);
        
        s1 = -konR0*A*D + koff*B;                               
        s2 = konR0*A*D - (koff + kin)*B + kout*C;               
        s3 = kin*B - (kout + kdeg)*C;                           
        s4 = koff*B + kq*E - (konR0*A + kp)*D;                  
        s5 = kg*(kp/kq) + kp*D - (kq + kg)*E;                   
        s = [s1; s2; s3; s4; s5];
    end

    % --- Initial Conditions ---
    function u0 = ic_sys(~)
        u0 = [0; 0; 0; D0; E0];
    end

    % --- Boundary Conditions ---
    function [pl, ql, pr, qr] = bc_sys(~, ~, ~, ur, ~)
        % Left (x=0): Constant Flux of A
        pl = [v_norm; 0; 0; 0; 0];
        ql = [1; 1; 1; 1; 1];
        % Right (x=xmax): A is completely absorbed 
        pr = [ur(1); 0; 0; 0; 0];
        qr = [0; 1; 1; 1; 1];
    end
end