% Generate synthetic data points
n = 100;  % Number of data points
m = 10;   % Number of inducing points

X = linspace(-5, 5, n)';  % n data points
Z = linspace(-5, 5, m)';  % m inducing points

% Define the RBF kernel function
sigma_f = 1.0;  % Kernel amplitude
l = 1.0;        % Length scale
kernel = @(x1, x2) sigma_f^2 * exp(-0.5 * pdist2(x1, x2).^2 / l^2);

% Compute kernel matrices
K_nn = kernel(X, X);      % Full kernel matrix (n x n)
K_mm = kernel(Z, Z);      % Kernel matrix for inducing points (m x m)
K_nm = kernel(X, Z);      % Kernel matrix between data and inducing points (n x m)

% Nyström approximation of the full kernel matrix
K_approx = K_nm * (K_mm \ K_nm');  % Approximate full kernel matrix

% Plot the full kernel and the approximated kernel
figure;
subplot(1,2,1);
imagesc(K_nn);
colorbar;
title('Full Kernel Matrix K');

subplot(1,2,2);
imagesc(K_approx);
colorbar;
title('Nyström Approximation');
