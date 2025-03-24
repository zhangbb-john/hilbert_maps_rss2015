% Number of inducing points
m = 40;
global scale
scale = 1;
% Define inducing points (we can uniformly sample or use a grid)
inducing_points = linspace(-10, 10, m + 1);  % Example 1D inducing points for simplicity


% Define the sparse kernel function
% sparse_kernel = @(x, x_prime) (norm(x - x_prime) < 1) * ((2 + cos(2 * pi * norm(x - x_prime))) ...
% 	/ 3 * (1 - norm(x - x_prime)) + ...
% 	(1 / (2 * pi)) * sin(2 * pi * norm(x - x_prime)));
sparse_kernel = @(x, x_prime) (norm(x - x_prime) < 1) * (1 - norm(x - x_prime));
x0 = 0;

x = rand(300, 1) * 10 - 5;
max_kernel1 = 0; max_kernel2 = 0;
figure;
colors = jet(length(inducing_points));

for i = 1 : length(inducing_points)
	samples = linspace(-10, 10, 10 * m + 1);
	vector = [];
	for j = 1 : length(samples)
		kv = sparse_kernel(samples(j), inducing_points(i));
		vector = [vector, kv];
	end
	plot(samples, vector, '-', 'color', colors(i,:)); hold on;
end
figure;
ratio = [];
for i = 1 : length(x)
	kernel1 = sparse_kernel(x0/scale, x(i)/scale);
	phi21 = basis_function(x0, inducing_points );
	phi22 = basis_function(x(i), inducing_points);
	kernel2 = phi21' * phi22;
	plot(x(i), kernel1, 'r.'); hold on;
	plot(x(i), kernel2, 'b.'); hold on;	
	if (kernel1 > max_kernel1)
		max_kernel1 = kernel1;
		max_kernel2 = kernel2;
	end
	if (kernel1 > 1e-3)
		ratio = [ratio, kernel2 / kernel1];
	else
		ratio = [ratio, 0];
	end
end
max_kernel1 / max_kernel2
figure;
plot(x, ratio, 'ro'); 

% Create feature matrix Phi for some x

% Compute kernel matrix K
K = zeros(m, m);
for i = 1:m
    for j = 1:m
        K(i, j) = sparse_kernel(inducing_points(i)/scale, inducing_points(j))/ scale;
    end
end

% Compute Phi' * Phi
Phi_phi = Phi' * Phi;
% Display the results
disp('Feature vector Phi:');
disp(Phi);
disp('Phi'' * Phi:');
disp(Phi_phi);
disp('Kernel matrix K:');
disp(K);

% Verify if Phi' * Phi equals the kernel matrix (up to a tolerance)
tolerance = 1e-6;
if norm(Phi_phi - K) < tolerance
    disp('Phi'' * Phi is approximately equal to the kernel matrix K.');
else
    disp('Phi'' * Phi is NOT equal to the kernel matrix K.');
end


function Phi = basis_function(x, inducing_points)
	global scale;
% 	sparse_kernel = @(x, x_prime) (norm(x - x_prime) < 1) * ((2 + cos(2 * pi * norm(x - x_prime))) / 3 * (1 - norm(x - x_prime)) + (1 / (2 * pi)) * sin(2 * pi * norm(x - x_prime)));
	sparse_kernel = @(x, x_prime) (norm(x - x_prime) < 1) * (1 - norm(x - x_prime));

	m = length(inducing_points);
	Phi = zeros(m, 1);  % Feature vector for a particular x (e.g., x = 0.5)
	for i = 1:m
	    Phi(i) = sparse_kernel(x/scale, inducing_points(i) / scale);
	end
	
end

