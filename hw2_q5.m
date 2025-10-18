rvals = linspace(0, 9, 1000);
eps = 1e-2;
figure(); hold on;
for r=rvals
    % A.y^3 + B.y^2 + C.y + D = 0.
    ode_coeffs = [-1 5 -r 1];
    y_ss_vec = roots(ode_coeffs);
    for j=1:length(y_ss_vec)
        % plot only real solutions.
        if (abs(imag(y_ss_vec(j))) < eps)
            plot(r, y_ss_vec(j), 'b.');
        end
    end
end
hold off;
title('Bifurcation Plot')
xlabel('r');
ylabel('y at steady state');
prettyfig;