% DD2424 Deep Learning - Assignment 2 - Hannes Kindbom

% Run this first: addpath Datasets/cifar-10-batches-mat/;

clear all
clc

% Read and preprocess data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

% -----------------------------
% Code for random search
filenames = {'data_batch_1.mat','data_batch_2.mat','data_batch_3.mat','data_batch_4.mat','data_batch_5.mat'};
%[X_train, Y_train, y_train, X_val, Y_val, y_val] = LoadAllBatches(filenames, 1000);
% -----------------------------

X_train_mean = mean(X_train, 2);
X_train_std = std(X_train, 0, 2);

X_train_norm = NormalizeMatrix(X_train, X_train_mean, X_train_std);
X_val_norm = NormalizeMatrix(X_val, X_train_mean, X_train_std);
X_test_norm = NormalizeMatrix(X_test, X_train_mean, X_train_std);

% Constants
K = size(Y_train, 1);
d = size(X_train, 1);

% Hyperparameters
nr_hidden_nodes = 50;
rng(400)
lambda = 0.5;
l_min = -3;
l_max = -2;
nr_trys = 10;
n_batch = 100;
eta_min = 0.00001;
eta_max = 0.1;
n_s = 2*floor(size(X_train_norm,2) / n_batch);
n_cycles = 1;
h = 0.0001;
GDparams = [n_batch, eta_min, eta_max, n_s, n_cycles];

% Initialize Parameters
[W, b] = InitParameters(nr_hidden_nodes, K, d);

% Random search
RandomSearch(l_min, l_max, X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, GDparams, W, b, nr_trys);

% For using smaller dimensions
sub_imgs = 2;
sub_dim = 20;
% Creating sub matrices
W{1} = W{1}(:,1:sub_dim);
X_train_norm = X_train_norm(1:sub_dim, 1:sub_imgs);
X_val_norm = X_val_norm(1:sub_dim, 1:sub_imgs);
X_test_norm = X_test_norm(1:sub_dim, 1:sub_imgs);
Y_train = Y_train(:, 1:sub_imgs);
y_train = y_train(1:sub_imgs, :);
Y_val = Y_val(:, 1:sub_imgs);
y_test = y_test(1:sub_imgs, :);
y_val = y_val(1:sub_imgs, :);
ComputeGradError(X_train_norm, Y_train, W, b, K, lambda, h);


% Training
disp('Training...');
[Wstar, bstar, val_acc] = MiniBatchGD(X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, GDparams, W, b, lambda);
disp('Finished training!');
accuracy = ComputeAccuracy(X_test_norm, y_test, Wstar, bstar);
disp('Accuracy:');
disp(accuracy);


% Definition of functions
function lambda_try = GetALambda(l_min, l_max)
l = l_min + (l_max - l_min)*rand(1, 1);
lambda_try = 10^l;
end

function RandomSearch(l_min, l_max, X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, GDparams, W, b, nr_trys)
disp('Searching for optimal Lambda');
val_accs = zeros(nr_trys,1);
lambdas = zeros(nr_trys,1);
for try_i = 1:nr_trys
    lambda_try = GetALambda(l_min, l_max);
    [~, ~, val_acc] = MiniBatchGD(X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, GDparams, W, b, lambda_try);
    val_accs(try_i) = val_acc(end);
    lambdas(try_i) = lambda_try;    
end
matlab.io.saveVariablesToScript('random_search.m',{'val_accs','lambdas'})
end

function acc = ComputeAccuracy(X, y, W, b)
y_pred = Predict(X, W, b);
y_diff = y_pred - y;
nr_wrong = nnz(y_diff);

acc = (length(y) - nr_wrong)/length(y);
end

function predictions = Predict(X, W, b)
n = size(X, 2);
predictions = zeros(n,1);

K = size(b{2},1);

for img = 1:n
    x = X(:, img);
    [p, ~] = EvaluateClassifier(x, W, b, K);
    [~, img_pred] = max(p);
    predictions(img) = img_pred;
end
end

function ComputeGradError(X, Y, W, b, K, lambda, h)
[P, H] = EvaluateClassifier(X, W, b, K);
[grad_b_an, grad_W_an] = ComputeGradients(X, Y, P, W, H, lambda);
[grad_b_num, grad_W_num] = ComputeGradsNumSlow(X, Y, W, b, lambda, h);

rel_grad_error_W1 = ComputeRelativeError(grad_W_an{1}, grad_W_num{1})
rel_grad_error_W2 = ComputeRelativeError(grad_W_an{2}, grad_W_num{2})
rel_grad_error_b1 = ComputeRelativeError(grad_b_an{1}, grad_b_num{1})
rel_grad_error_b2 = ComputeRelativeError(grad_b_an{2}, grad_b_num{2})

end

function [X, Y, y, X_val, Y_val, y_val] = LoadAllBatches(filenames, n_val)
X = [];
Y = [];
y = [];
for batch_nr = 1:length(filenames)
    batch_filename = filenames{batch_nr};
    [X_batch, Y_batch, y_batch] = LoadBatch(batch_filename);
    X = horzcat(X, X_batch);
    Y = horzcat(Y, Y_batch);
    y = vertcat(y, y_batch);
end
n = size(X, 2);
X_val = X(:, n-n_val+1:n);
Y_val = Y(:, n-n_val+1:n);
y_val = y(n-n_val+1:n, :);

X = X(:, 1:n-n_val);
Y = Y(:, 1:n-n_val);
y = y(1:n-n_val, :);
end

function [X, Y, y] = LoadBatch(filename)
batch = load(filename);

X = transpose(double(batch.data));
y = double(batch.labels) + 1;

onehot = @(label_vec)bsxfun(@eq, label_vec(:), 1:max(label_vec));
Y = transpose(onehot(y));
end

function X_norm = NormalizeMatrix(X, X_mean, X_std)
X = X - repmat(X_mean, [1, size(X, 2)]);
X_norm = X ./ repmat(X_std, [1, size(X, 2)]);
end

function [W, b] = InitParameters(m, K, d)
w1_std = 1/sqrt(d);
w2_std = 1/sqrt(m);

W1 = w1_std * randn(m, d);
W2 = w2_std * randn(K, m);
b1 = zeros(m, 1);
b2 = zeros(K, 1);

W = {W1, W2};
b = {b1, b2};
end

function [grad_b, grad_W] = ComputeGradients(X, Y, P, W, H, lambda)
n = size(X, 2);

G = -(Y - P);
dL_dW2 = G*transpose(H)/n;
dL_db2 = G*ones(n, 1)/n;

G = transpose(W{2})*G;
G = G.*(H > 0);  

dL_dW1 = G*transpose(X)/n;
dL_db1 = G*ones(n, 1)/n;

grad_W  = {dL_dW1 + 2*lambda*W{1}, dL_dW2 + 2*lambda*W{2}};
grad_b = {dL_db1, dL_db2};
end

function relative_error = ComputeRelativeError(grad_an, grad_num)
eps = 0.0001;
relative_error = norm(grad_an - grad_num) / max([eps, norm(grad_an) + norm(grad_num)]);
end

function eta = GetCLR(t, eta_min, eta_max, n_s)
cycle = floor(1+t/(2*n_s));
x = abs(t/n_s - 2*cycle + 1);
eta = eta_min + (eta_max-eta_min)*max(0, (1-x));
end

function [batch_start, batch_end] = GetBatchRange(iter, n_batch, n)
iter = mod(iter, n/n_batch);
if iter == 0
    iter = n/n_batch;
end
batch_start = (iter-1)*n_batch + 1;
batch_end = iter*n_batch;
end

function [Wstar, bstar, val_acc] = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, GDparams, W, b, lambda)
n = size(X, 2);
K = size(Y, 1);
n_batch = GDparams(1);
eta_min = GDparams(2);
eta_max = GDparams(3);
n_s = GDparams(4);
n_cycles = GDparams(5);
train_cost = zeros(n_cycles*10,1);
val_cost = zeros(n_cycles*10,1);
val_acc = zeros(n_cycles*10,1);
train_acc = zeros(n_cycles*10,1);
etas = zeros(n_cycles*10,1);
tenx_t = 0;
nr_upds = n_cycles*2*n_s;

for t = 1:nr_upds
    eta = GetCLR(t, eta_min, eta_max, n_s);
    [batch_start, batch_end] = GetBatchRange(t, n_batch, n);

    X_batch = X(:, batch_start:batch_end);
    Y_batch = Y(:, batch_start:batch_end);

    [P, H] = EvaluateClassifier(X_batch, W, b, K);
    [grad_b, grad_W] = ComputeGradients(X_batch, Y_batch, P, W, H, lambda);

    W{1} = W{1} - eta * grad_W{1};
    W{2} = W{2} - eta * grad_W{2};
    b{1} = b{1} - eta * grad_b{1};
    b{2} = b{2} - eta * grad_b{2};
    
    %PLot 10 times per cycle
    if mod(t, nr_upds/n_cycles/10) == 0 || t == 1
        tenx_t = tenx_t + 1;
        disp(tenx_t);
        train_cost(tenx_t) = ComputeCost(X, Y, W, b, lambda);
        val_cost(tenx_t) = ComputeCost(X_val, Y_val, W, b, lambda);
        val_acc(tenx_t) = ComputeAccuracy(X_val, y_val, W, b);
        train_acc(tenx_t) = ComputeAccuracy(X, y, W, b);
        etas(tenx_t) = eta;
    end
end
upd_steps = linspace(0,nr_upds,n_cycles*10+1);
figure(1)
plot(upd_steps, train_cost);
title('Training Cost Progress')
xlabel('Upd Steps')
ylabel('Cost')
hold on
plot(upd_steps, val_cost);
legend('train','val')
hold off

figure(2)
plot(upd_steps, etas)
title('Cyclical learning rate')
xlabel('Upd Steps')
ylabel('Eta')

figure(3)
plot(upd_steps, train_acc);
title('Training Accuracy Progress')
xlabel('Upd Steps')
ylabel('Accuracy')
hold on
plot(upd_steps, val_acc);
legend('train','val')
hold off

Wstar = W;
bstar = b;
end

function [P, H] = EvaluateClassifier(X, W, b, K)
n = size(X, 2);

s1 = W{1}*X + b{1};
H = max(zeros(size(W{1}, 1), n), s1);
s = W{2}*H + b{2};

P = zeros(K, n);
for col = 1:n
    P(:,col) = Softmax(s(:,col));
end
end

function P = Softmax(s)
P = exp(s)./(transpose(ones(size(s)))*exp(s));
end

function J = ComputeCost(X, Y, W, b, lambda)
n = size(X, 2);

J_loss = 0;
J_regul = 0;

for img = 1:n
    x = X(:,img);
    y = Y(:,img);
    J_loss = J_loss + ComputeLCross(x, y, W, b);
end
J_loss = J_loss/n;

for w_mat = 1:length(W)
    for row = 1:size(W{w_mat},1)
        for col = 1:size(W{w_mat},2)     
            J_regul = J_regul + W{w_mat}(row, col)^2;
        end    
    end
end

J_regul = lambda*J_regul;
J = J_loss + J_regul;
end
function LCross = ComputeLCross(x, y, W, b)
K = size(y, 1);
[p, ~] = EvaluateClassifier(x, W, b, K);
LCross = -log(transpose(y)*p);
end




% Functions copied from problem instructions
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end