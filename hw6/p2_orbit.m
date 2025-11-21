%% Realistic 2D phase portrait with two spirals, a saddle, and periodic orbit
clear; clc; close all;

% Grid for vector field
[x,y] = meshgrid(linspace(-4,4,30), linspace(-4,4,30));

% Parameters for synthetic vector field
% Two spirals at (-1,0) and (1,0), saddle at (0,0)
S1 = [-1,0]; S2 = [1,0]; Sad = [0,0];

% Define synthetic vector field
u = -(x-S1(1)) - (y-S1(2));  % attraction to S1
v = (x-S1(1)) - (y-S1(2));

u2 = -(x-S2(1)) - (y-S2(2)); % attraction to S2
v2 = (x-S2(1)) - (y-S2(2));

u_saddle = (x-Sad(1)) .* (1-(x.^2+y.^2)/8); % unstable along x
v_saddle = -(y-Sad(2)) .* (1-(x.^2+y.^2)/8); % stable along y

% Combine vector fields (weighted)
u_total = u.*exp(-((x+1).^2+(y).^2)) + u2.*exp(-((x-1).^2+y.^2)) + u_saddle;
v_total = v.*exp(-((x+1).^2+(y).^2)) + v2.*exp(-((x-1).^2+y.^2)) + v_saddle;

% Plot vector field
figure; hold on; axis equal; grid on
quiver(x,y,u_total,v_total,1,'Color',[0.7 0.7 0.7])

% Plot stable spirals and saddle
plot(S1(1), S1(2), 'bo', 'MarkerFaceColor','b', 'MarkerSize',8)
text(S1(1)+0.2,S1(2),'S_1','FontSize',12)
plot(S2(1), S2(2), 'bo', 'MarkerFaceColor','b', 'MarkerSize',8)
text(S2(1)+0.2,S2(2),'S_2','FontSize',12)
plot(Sad(1), Sad(2), 'ks', 'MarkerFaceColor','k', 'MarkerSize',8)
text(Sad(1)+0.2,Sad(2),'Saddle','FontSize',12)

% Outer periodic orbit (approximate)
theta = linspace(0,2*pi,200);
R = 3; xc = 0; yc = 0;
xP = xc + R*cos(theta);
yP = yc + R*sin(theta);
plot(xP, yP, 'r', 'LineWidth',1.5)

% Plot sample trajectories
traj_ICs = [-2,1; -2,-1; 2,1; 2,-1; 0,2; 0,-2];
tspan = 0:0.05:20;
for k = 1:size(traj_ICs,1)
    [t,Y] = ode45(@(t,Y) [interp2(x,y,u_total,Y(1),Y(2)); ...
                           interp2(x,y,v_total,Y(1),Y(2))], tspan, traj_ICs(k,:));
    plot(Y(:,1),Y(:,2),'b','LineWidth',1.2)
    plot(Y(1,1),Y(1,2),'go','MarkerFaceColor','g') % initial
    plot(Y(end,1),Y(end,2),'ko','MarkerFaceColor','k') % final
end

xlabel('x'); ylabel('y'); title('Phase portrait with two spirals, saddle, and periodic orbit')
legend('Vector field','S_1','S_2','Saddle','Periodic orbit','Trajectory','Location','best')
