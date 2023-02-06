%------------------------------------------------------------------------------%
% Objective: f_i(x) = poly_i(4), a_{i1} > 0
% Stochastic gradient: g_i(x) = \nabla f_i(x) + z_i, z_i ~ Unif(-5, 5)
%------------------------------------------------------------------------------%

clear;

% distributed setting
N = 2; %number of clients
range = 1;
H = 1;
h = linspace(-range, 2*range, N)';
Alocal = [ones(N, 1), -3*ones(N, 1), -H*h, ones(N, 1)]; %local coefficient matrix, N by 2
a = mean(Alocal, 1); % global coefficient


% hyper parameters
L0 = 1;
A = 1;
sigma = 1;
if a(2) > 0
    x0 = -1;
else
    x0 = 1;
end
%eta = 10*N * epsilon^2/sigma^2/(A*L0);
R_max = 500;
K = 8; % number of skipped communication
tau = 5; % tau = gamma/eta
eta = 1e-2;

% Local clipping
gamma = tau*eta;
xr = x0;
loss_local = zeros(R_max+1, 1);
loss_local(1) = (a(1)*x0^4 + a(2)*x0^3 + a(3)*x0^2 + a(4)*x0);
for r = 1:R_max
    xr_delta = 0;
    clip_num_i = 0;
    for i = 1:N
        xk = xr;
        % local updates
        for k = 1:K
            Gk = 4*Alocal(i, 1)*xk^3 + 3*Alocal(i, 2)*xk^2 + 2*Alocal(i, 3)*xk + Alocal(i, 4);
            gk = Gk + (sigma - 2*sigma*rand(1)); %inject noise
            if abs(gk) <= gamma/eta
                xk = xk - eta*gk;
            else
                xk = xk - gamma*gk/abs(gk);
                clip_num_i = clip_num_i + 1;
            end
        end
        xr_delta = xr_delta + xk - xr; %record local increments
    end
    clip_freq = clip_num_i/(K*N);
    xr = xr + xr_delta/N; %synchronization
    Gr = 4*a(1)*xr^3 + 3*a(2)*xr^2 + 2*a(3)*xr + a(4);

    loss_local(r+1) = (a(1)*xr^4 + a(2)*xr^3 + a(3)*xr^2 + a(4)*xr);
end
x_local = xr;

% Episode
gamma = tau*eta;
xr = x0;
clip_num_1 = 0;
loss_episode = zeros(R_max+1, 1);
loss_episode(1) = (a(1)*x0^4 + a(2)*x0^3 + a(3)*x0^2 + a(4)*x0);
for r = 1:R_max
    % compute Gr and Gri
    Gri = zeros(N, 1);
    for i = 1:N
        Gri(i) = 4*Alocal(i, 1)*xr^3 + 3*Alocal(i, 2)*xr^2 + 2*Alocal(i, 3)*xr + Alocal(i, 4) + (sigma - 2*sigma*rand(1));
    end
    Gr = mean(Gri);

    if abs(Gr) <= gamma/eta
       % local updates without clipping
        xr_delta = 0;
        for i = 1:N
            xk = xr;
            for k = 1:K
                Gk = 4*Alocal(i, 1)*xk^3 + 3*Alocal(i, 2)*xk^2 + 2*Alocal(i, 3)*xk + Alocal(i, 4); % true local gradient
                gk = Gk + (sigma - 2*sigma*rand(1)); %stochstic gradient
                xk = xk - eta*(gk - Gri(i) + Gr);
            end
            xr_delta = xr_delta + xk - xr; %record local increments
        end
    else
        clip_num_1 = clip_num_1 + 1;
        % local updates without clipping
        xr_delta = 0;
        for i = 1:N
            xk = xr;
            for k = 1:K
                Gk = 4*Alocal(i, 1)*xk^3 + 3*Alocal(i, 2)*xk^2 + 2*Alocal(i, 3)*xk + Alocal(i, 4);% true local gradient
                gk = Gk + (sigma - 2*sigma*rand(1)); %stochstic gradient
                xk = xk - gamma*(gk - Gri(i) + Gr)/abs(gk - Gri(i) + Gr);
            end
            xr_delta = xr_delta + xk - xr; %record local increments
        end
    end

    xr = xr + xr_delta/N; %synchronization 
    loss_episode(r+1) = (a(1)*xr^4 + a(2)*xr^3 + a(3)*xr^2 + a(4)*xr);

    if mod(r, 1e3)==0
        fprintf('The current loss: %f \n', loss_episode(r));
    end
end
x_episode = xr;




% Mini-batch SGD with clipping
%eta = K*eta;
% gamma = tau*eta;
% xt = x0;
% clip_num = 0;
% loss_mini = zeros(R_max+1, 1);
% loss_mini(1) = (a(1)*x0^4 + a(2)*x0^3 + a(3)*x0^2 + a(4)*x0);
% for t = 1:R_max
%     Gt = 4*a(1)*xt^3 + 3*a(2)*xt^2 + 2*a(3)*xt + a(4);
%     gt = Gt + mean(sigma - 2*sigma*rand(K, 1));
%     if abs(gt) <= gamma/eta
%         xt = xt - eta*gt;
%     else
%         xt = xt - gamma*gt/abs(gt);
%         clip_num = clip_num + 1;
%     end

%     loss_mini(t+1) = (a(1)*xt^4 + a(2)*xt^3 + a(3)*xt^2 + a(4)*xt);
% end
% x_mini = xt;

loss_matrix = [loss_local loss_episode];
solutions = [x_local x_episode];
%clip_plot(loss_matrix(1:31,:), Alocal, solutions, 'clip_h4.eps');
legend_ind = 1;
if(H == 1)
    legend_ind = 1;
end
trajectory_plot(loss_matrix(1:31,:), 'traj_h1.eps', legend_ind);
objective_plot(Alocal, solutions, 'obj_h1.eps', legend_ind);

