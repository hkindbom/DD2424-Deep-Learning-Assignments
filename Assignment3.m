% DD2424 Deep Learning - Assignment 3 - Hannes Kindbom

% Run this first: addpath Datasets/cifar-10-batches-mat/;

clear all
clc
use_sub_dim_data = false;
use_all_data = true;
do_random_search = false;
use_batch_norm = false;

% Read and preprocess data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

if use_all_data
filenames = {'data_batch_1.mat','data_batch_2.mat','data_batch_3.mat','data_batch_4.mat','data_batch_5.mat'};
[X_train, Y_train, y_train, X_val, Y_val, y_val] = LoadAllBatches(filenames, 1000);
end

X_train_mean = mean(X_train, 2);
X_train_std = std(X_train, 0, 2);

X_train_norm = NormalizeMatrix(X_train, X_train_mean, X_train_std);
X_val_norm = NormalizeMatrix(X_val, X_train_mean, X_train_std);
X_test_norm = NormalizeMatrix(X_test, X_train_mean, X_train_std);

% Constants
K = size(Y_train, 1);
d = size(X_train, 1);

% Hyperparameters
hidden_layers_nodes = [50, 50];
rng(400)
lambda = 0.005;
l_min = -2;
l_max = -1;
nr_trys = 8;
n_batch = 100;
eta_min = 0.00001;
eta_max = 0.1;
n_s = 5 * 45000/n_batch;
n_cycles = 2;
h = 0.0001;
GDparams = [n_batch, eta_min, eta_max, n_s, n_cycles];

% For using smaller dimensions
if use_sub_dim_data
    sub_imgs = 10;
    sub_dim = 200;
    d = sub_dim;
    % Creating sub matrices
    X_train_norm = X_train_norm(1:sub_dim, 1:sub_imgs);
    X_val_norm = X_val_norm(1:sub_dim, 1:sub_imgs);
    X_test_norm = X_test_norm(1:sub_dim, 1:sub_imgs);
    Y_train = Y_train(:, 1:sub_imgs);
    y_train = y_train(1:sub_imgs, :);
    Y_val = Y_val(:, 1:sub_imgs);
    y_test = y_test(1:sub_imgs, :);
    y_val = y_val(1:sub_imgs, :);
end

% Initialize Parameters
NetParams = InitParameters(hidden_layers_nodes, K, d);

if use_batch_norm
    NetParams.use_bn = true;
else
    NetParams.use_bn = false;
end

% Random search
if do_random_search
    RandomSearch(l_min, l_max, X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, GDparams, NetParams, nr_trys);
end

%ComputeGradError(X_train_norm, Y_train, NetParams, K, lambda, h);

% Training
disp('Training...');
[NetParamsStar, val_acc, mu_av, var_av] = MiniBatchGD(X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, GDparams, NetParams, lambda);
disp('Finished training!');
if NetParams.use_bn
    accuracy = ComputeAccuracy(X_test_norm, y_test, NetParamsStar, mu_av, var_av);
else
    accuracy = ComputeAccuracy(X_test_norm, y_test, NetParamsStar);
end
disp('Accuracy:');
disp(accuracy);


% Definition of functions
function lambda_try = GetALambda(l_min, l_max)
l = l_min + (l_max - l_min)*rand(1, 1);
lambda_try = 10^l;
end

function RandomSearch(l_min, l_max, X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, GDparams, NetParams, nr_trys)
disp('Searching for optimal Lambda');
val_accs = zeros(nr_trys,1);
lambdas = zeros(nr_trys,1);
for try_i = 1:nr_trys
    lambda_try = GetALambda(l_min, l_max);
    [~, val_acc, ~, ~] = MiniBatchGD(X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, GDparams, NetParams, lambda_try);
    val_accs(try_i) = val_acc(end);
    lambdas(try_i) = lambda_try;    
end
matlab.io.saveVariablesToScript('random_search.m',{'val_accs','lambdas'})
end

function acc = ComputeAccuracy(X, y, NetParams, varargin)
y_pred = Predict(X, NetParams, varargin);
y_diff = y_pred - y;
nr_wrong = nnz(y_diff);

acc = (length(y) - nr_wrong)/length(y);
end

function predictions = Predict(X, NetParams, varargin)
n = size(X, 2);
predictions = zeros(n,1);

K = length(NetParams.b{end});
[P, ~, ~] = EvaluateClassifier(X, NetParams, K, varargin);
for img = 1:n
    [~, img_pred] = max(P(:,img));
    predictions(img) = img_pred;
end
end

function ComputeGradError(X, Y, NetParams, K, lambda, h)
[P, H, BNData] = EvaluateClassifier(X, NetParams, K);
Grads_an = ComputeGradients(X, Y, P, NetParams, BNData, H, lambda);
Grads_num = ComputeGradsNumSlow(X, Y, NetParams, lambda, h);

for l = 1:length(Grads_an.W)
    rel_grad_error_W = ComputeRelativeError(Grads_an.W{l}, Grads_num.W{l})
    rel_grad_error_b = ComputeRelativeError(Grads_an.b{l}, Grads_num.b{l})
end
if NetParams.use_bn
    for l = 1:length(Grads_an.gammas)
        rel_grad_error_gammas = ComputeRelativeError(Grads_an.gammas{l}, Grads_num.gammas{l})
        rel_grad_error_betas = ComputeRelativeError(Grads_an.betas{l}, Grads_num.betas{l})
    end
end
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

% Using He initialization std = sqrt(2/prev_nr_nodes);
function NetParams = InitParameters(hidden_layers_nodes, K, d)
nr_layers = length(hidden_layers_nodes) + 1;
W = cell(1, nr_layers);
b = cell(1, nr_layers);
betas = cell(1, nr_layers-1);
gammas = cell(1, nr_layers-1);

for layer = 1:nr_layers
    if layer == 1
        w_std = sqrt(2/d);
        W{1} = w_std * randn(hidden_layers_nodes(1), d);
        b{1} = zeros(hidden_layers_nodes(1), 1);
        betas{1} = zeros(hidden_layers_nodes(1), 1);
        gammas{1} = ones(hidden_layers_nodes(1), 1);
    elseif layer == nr_layers
        w_std = sqrt(2/hidden_layers_nodes(layer-1));
        W{nr_layers} = w_std * randn(K, hidden_layers_nodes(end));
        b{nr_layers} = zeros(K, 1);
    else 
        w_std = sqrt(2/hidden_layers_nodes(layer-1));
        W{layer} = w_std * randn(hidden_layers_nodes(layer), hidden_layers_nodes(layer-1));
        b{layer} = zeros(hidden_layers_nodes(layer), 1);
        betas{layer} = zeros(hidden_layers_nodes(layer), 1);
        gammas{layer} = ones(hidden_layers_nodes(layer), 1);
    end
end
NetParams.W = W;
NetParams.b = b;
NetParams.betas = betas;
NetParams.gammas = gammas;
end

function Grads = ComputeGradients(X, Y, P, NetParams, BNData, H, lambda)
n = size(X, 2);
k = length(NetParams.W);
Grads.W = cell(numel(NetParams.W), 1);
Grads.b = cell(numel(NetParams.b), 1);
Grads.gammas = cell(numel(NetParams.gammas), 1);
Grads.betas = cell(numel(NetParams.betas), 1);

G = -(Y - P);
if ~NetParams.use_bn
    for l=k:-1:2
     dL_dWl = G*transpose(H{l-1})/n;
     dL_dbl = G*ones(n, 1)/n;
     G = transpose(NetParams.W{l})*G;
     G = G.*(H{l-1} > 0);  

     Grads.W{l} = dL_dWl + 2*lambda*NetParams.W{l};
     Grads.b{l} = dL_dbl;
    end
    Grads.W{1} = G*transpose(X)/n + 2*lambda*NetParams.W{1};
    Grads.b{1} = G*ones(n, 1)/n;
else   
    Grads.W{k} = G*transpose(H{k-1})/n + 2*lambda*NetParams.W{k};
    Grads.b{k} = G*ones(n, 1)/n;
    G = transpose(NetParams.W{k})*G;
    G = G.*(H{k-1} > 0);
    for l=k-1:-1:1
        Grads.gammas{l} = (G .* BNData.S_hat{l}) * ones(n, 1)/n;
        Grads.betas{l} = G*ones(n, 1)/n;        
        G = G .* (NetParams.gammas{l}*ones(1, n));
        G = BatchNormBackPass(G, BNData.S{l}, BNData.mu{l}, BNData.vari{l});
        
        if l > 1
        Grads.W{l} = G*transpose(H{l-1})/n + 2*lambda*NetParams.W{l};
        Grads.b{l} = G*ones(n, 1)/n;
        G = transpose(NetParams.W{l})*G;
        G = G.*(H{l-1} > 0);  
        else
            Grads.W{1} = G*transpose(X)/n + 2*lambda*NetParams.W{1};
            Grads.b{1} = G*ones(n, 1)/n;
        end
    end
end
end
function G = BatchNormBackPass(G, S_i, mu_i, var_i)
n = size(G, 2);
sigma_1 = transpose((var_i + eps).^(-0.5));
sigma_2 = transpose((var_i + eps).^(-1.5));
G_1 = G .* (sigma_1*ones(1, n));
G_2 = G .* (sigma_2*ones(1, n));

D = S_i - mu_i*ones(1, n);
c = (G_2 .* D)*ones(n, 1);
G = G_1 - (1/n)*(G_1*ones(n, 1))*ones(1, n) - (1/n)*D.*(c*ones(1, n));
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

function NetParams = UpdateNetParams(Grads, NetParams, eta)
for l = 1:length(NetParams.W)
    NetParams.W{l} = NetParams.W{l} - eta * Grads.W{l};
    NetParams.b{l} = NetParams.b{l} - eta * Grads.b{l};    
end
if NetParams.use_bn
    for l = 1:length(NetParams.gammas)
        NetParams.gammas{l} = NetParams.gammas{l} - eta * Grads.gammas{l};
        NetParams.betas{l} = NetParams.betas{l} - eta * Grads.betas{l};
    end
end
end
function [X, Y, y] = ShuffleData(X, Y, y)
n = size(X, 2);
new_order = randperm(n);
% Apply that order to all arrays.
X = X(:, new_order);
Y = Y(:, new_order);
y = y(new_order, :);
end
function [mu_av, var_av] = UpdateMovAvEst(mu_av, var_av, BNData)
alpha = 0.5;
for layer = 1:length(mu_av)
    if length(mu_av{layer}) < 1
        mu_av{layer} = BNData.mu{layer};
        var_av{layer} = BNData.vari{layer};
    else
        mu_av{layer} = alpha * mu_av{layer} + (1-alpha) * BNData.mu{layer};
        var_av{layer} = alpha * var_av{layer} + (1-alpha) * BNData.vari{layer};
    end
end
end

function [NetParamsStar, val_acc, mu_av, var_av] = MiniBatchGD(X, Y, y, X_val, Y_val, y_val, GDparams, NetParams, lambda)
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

mu_av = cell(1, length(NetParams.W)-1);
var_av = cell(1, length(NetParams.W)-1);

for t = 1:nr_upds
    eta = GetCLR(t, eta_min, eta_max, n_s);
    [batch_start, batch_end] = GetBatchRange(t, n_batch, n);
    
    % Suffle data after each epoch
    if mod(t*n_batch, n) == 0
        [X, Y, y] = ShuffleData(X, Y, y);
    end

    X_batch = X(:, batch_start:batch_end);
    Y_batch = Y(:, batch_start:batch_end);

    [P, H, BNData] = EvaluateClassifier(X_batch, NetParams, K);    
    Grads = ComputeGradients(X_batch, Y_batch, P, NetParams, BNData, H, lambda);

    NetParams = UpdateNetParams(Grads, NetParams, eta);
    if NetParams.use_bn
        [mu_av, var_av] = UpdateMovAvEst(mu_av, var_av, BNData);
    end
    
    %Plot 10 times per cycle
    if mod(t, nr_upds/n_cycles/10) == 0 || t == 1
        tenx_t = tenx_t + 1;
        disp(tenx_t);
        train_cost(tenx_t) = ComputeCost(X, Y, NetParams, lambda, mu_av, var_av);
        val_cost(tenx_t) = ComputeCost(X_val, Y_val, NetParams, lambda, mu_av, var_av);
        val_acc(tenx_t) = ComputeAccuracy(X_val, y_val, NetParams, mu_av, var_av);
        train_acc(tenx_t) = ComputeAccuracy(X, y, NetParams, mu_av, var_av);
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

NetParamsStar = NetParams;
end

function s_hat_i = BatchNormailize(s_i, mu_s_i, var_s_i)
s_hat_i = (diag(var_s_i + eps))^(-1/2) * (s_i - mu_s_i);
end
function [P, H, BNData] = EvaluateClassifier(X, NetParams, K, varargin)
n = size(X, 2);
k = length(NetParams.W);
H = cell(1, k-1);
S = cell(1, k-1);
S_hat = cell(1, k-1);
mu = cell(1, k-1);
vari = cell(1, k-1);

for i = 1:k-1
    s_i = NetParams.W{i}*X + NetParams.b{i};
   
    if NetParams.use_bn
        mu_s_i = mean(s_i, 2);
        var_s_i = var(s_i, 0, 2) * (n-1) / n;
        
        % If using validation or test data
        if nargin > 4
            s_hat_i = BatchNormailize(s_i, varargin{1}{1}{1}{i}, varargin{1}{1}{2}{i});
        else
            s_hat_i = BatchNormailize(s_i, mu_s_i, var_s_i);
        end
        
        s_tilde_i = NetParams.gammas{i} .* s_hat_i + NetParams.betas{i};
        X = max(0, s_tilde_i);
        
        S_hat{i} = s_hat_i;
        mu{i} = mu_s_i;
        vari{i} = transpose(var_s_i);
    else
        X = max(0, s_i);
    end
    S{i} = s_i;
    H{i} = X;
end

s_k = NetParams.W{k}*X + NetParams.b{k};
P = zeros(K, n);
for col = 1:n
    P(:,col) = Softmax(s_k(:,col));
end
BNData.S = S;
BNData.S_hat = S_hat;
BNData.mu = mu;
BNData.vari = vari;
end

function P = Softmax(s)
P = exp(s)./(transpose(ones(size(s)))*exp(s));
end

function J = ComputeCost(X, Y, NetParams, lambda, varargin)
n = size(X, 2);

J_regul = 0;
J_loss =  ComputeLCross(X, Y, NetParams, varargin)/n;

for w_mat = 1:length(NetParams.W)
    J_regul = J_regul + sum(NetParams.W{w_mat}.^2, 'all');
end

J_regul = lambda*J_regul;
J = J_loss + J_regul;
end
function LCross = ComputeLCross(X, Y, NetParams, varargin)
K = size(Y, 1);
[P, ~, ~] = EvaluateClassifier(X, NetParams, K, varargin);
LCross = 0;
for img = 1:size(Y,2)
    y = Y(:,img);
    p = P(:,img);
    LCross = LCross -log(transpose(y)*p);
end
end



% Functions copied from problem instructions
function Grads = ComputeGradsNumSlow(X, Y, NetParams, lambda, h)

Grads.W = cell(numel(NetParams.W), 1);
Grads.b = cell(numel(NetParams.b), 1);
if NetParams.use_bn
    Grads.gammas = cell(numel(NetParams.gammas), 1);
    Grads.betas = cell(numel(NetParams.betas), 1);
end

for j=1:length(NetParams.b)
    Grads.b{j} = zeros(size(NetParams.b{j}));
    NetTry = NetParams;
    for i=1:length(NetParams.b{j})
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) - h;
        NetTry.b = b_try;
        c1 = ComputeCost(X, Y, NetTry, lambda);        
        
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) + h;
        NetTry.b = b_try;        
        c2 = ComputeCost(X, Y, NetTry, lambda);
        
        Grads.b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(NetParams.W)
    Grads.W{j} = zeros(size(NetParams.W{j}));
        NetTry = NetParams;
    for i=1:numel(NetParams.W{j})
        
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) - h;
        NetTry.W = W_try;        
        c1 = ComputeCost(X, Y, NetTry, lambda);
    
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) + h;
        NetTry.W = W_try;        
        c2 = ComputeCost(X, Y, NetTry, lambda);
    
        Grads.W{j}(i) = (c2-c1) / (2*h);
    end
end

if NetParams.use_bn
    for j=1:length(NetParams.gammas)
        Grads.gammas{j} = zeros(size(NetParams.gammas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.gammas{j})
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) - h;
            NetTry.gammas = gammas_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda);
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) + h;
            NetTry.gammas = gammas_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda);
            
            Grads.gammas{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    for j=1:length(NetParams.betas)
        Grads.betas{j} = zeros(size(NetParams.betas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.betas{j})
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) - h;
            NetTry.betas = betas_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda);
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) + h;
            NetTry.betas = betas_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda);
            
            Grads.betas{j}(i) = (c2-c1) / (2*h);
        end
    end    
end
end