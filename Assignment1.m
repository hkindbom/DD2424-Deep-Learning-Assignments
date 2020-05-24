% DD2424 Deep Learning - Assignment 1 - Hannes Kindbom

% Run this first: addpath DeepLearningAssignments/Datasets/cifar-10-batches-mat/;

clear all
clf
clc

% Hyperparameters
rng(400)
lambda = 1;
n_batch = 100;
eta = 0.001;
n_epochs = 40;
h = 0.0001;

% Read and preprocess data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');


X_train_mean = mean(X_train, 2);
X_train_std = std(X_train, 0, 2);

X_train_norm = NormalizeMatrix(X_train, X_train_mean, X_train_std);
X_val_norm = NormalizeMatrix(X_val, X_train_mean, X_train_std);
X_test_norm = NormalizeMatrix(X_test, X_train_mean, X_train_std);

% Initialize W and b according to N(0, 0.01)
K = size(Y_train, 1);
d = size(X_train, 1);
weights_std = 0.01;
W = weights_std * randn(K, d);
b = weights_std * randn(K, 1);

% Testing gradients
sub_imgs = 10;
sub_dim = 10;
P_testing = EvaluateClassifier(X_train_norm(1:sub_dim, 1:sub_imgs), W(:,1:sub_dim), b, K);
[grad_W, grad_b] = ComputeGradients(X_train_norm(1:sub_dim, 1:sub_imgs), Y_train(:, 1:sub_imgs), P_testing, W(:,1:sub_dim), lambda);
[grad_b_num, grad_W_num] = ComputeGradsNumSlow(X_train_norm(1:sub_dim, 1:sub_imgs), Y_train(:, 1:sub_imgs), W(:,1:sub_dim), b, lambda, h);

rel_error_W = ComputeRelativeError(grad_W, grad_W_num);
rel_error_b = ComputeRelativeError(grad_b, grad_b_num);
disp('relative error on W gradient:'); 
disp(rel_error_W);
disp('relative error on b gradient:'); 
disp(rel_error_b);


% Training
disp('Training...');
GDparams = [n_batch, eta, n_epochs];
[Wstar, bstar] = MiniBatchGD(X_train_norm, Y_train, X_val_norm, Y_val, GDparams, W, b, lambda);
disp('Finished training!');
accuracy = ComputeAccuracy(X_test_norm, y_test, Wstar, bstar);
disp('Accuracy:');
disp(accuracy);

PlotW(Wstar);

% Definition of functions
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

function P = EvaluateClassifier(X, W, b, K)
n = size(X, 2);
s = W*X + b;
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
K = size(Y, 1);
d = size(X, 1);

J_loss = 0;
J_regul = 0;

for img = 1:n
    x = X(:,img);
    y = Y(:,img);
    J_loss = J_loss + ComputeLCross(x, y, W, b);
end
J_loss = J_loss/n;

for row = 1:K
    for col = 1:d
        J_regul = J_regul + W(row, col)^2;
    end    
end
J_regul = lambda*J_regul;
J = J_loss + J_regul;
end

function LCross = ComputeLCross(x, y, W, b)
p = Softmax(W*x + b);
LCross = -log(transpose(y)*p);
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

for img = 1:n
    x = X(:, img);
    p = Softmax(W*x + b);
    [max_prob, img_pred] = max(p);
    predictions(img) = img_pred;
end
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
n = size(X, 2);
G = -(Y - P);

grad_W = G*transpose(X)./n + 2*lambda*W;
grad_b = G*ones(n, 1)/n;

end

function relative_error = ComputeRelativeError(grad_an, grad_num)
eps = 0.0001;
relative_error = norm(grad_an - grad_num) / max([eps, norm(grad_an) + norm(grad_num)]);
end

function [Wstar, bstar] = MiniBatchGD(X, Y, X_val, Y_val, GDparams, W, b, lambda)
n = size(X, 2);
K = size(Y, 1);
n_batch = GDparams(1);
eta = GDparams(2);
n_epochs = GDparams(3);
train_cost = zeros(n_epochs,1);
val_cost = zeros(n_epochs,1);

for i = 1:n_epochs
    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        X_batch = X(:, j_start:j_end);
        Y_batch = Y(:, j_start:j_end);

        P = EvaluateClassifier(X_batch, W, b, K);
        [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, P, W, lambda);

        W = W - eta * grad_W;
        b = b - eta * grad_b;
    end
    train_cost(i) = ComputeCost(X, Y, W, b, lambda);
    val_cost(i) = ComputeCost(X_val, Y_val, W, b, lambda);
end
figure(1)
plot(train_cost);
title('Training Progress')
xlabel('Epochs')
ylabel('Loss')
hold on
plot(val_cost);
legend('train','val')
hold off

Wstar = W;
bstar = b;
end

function PlotW(W)
K = size(W, 1);
s_im = cell(K);
for i=1:K
    im = reshape(W(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure(2)
for class = 1:K
    subplot(1, K, class), imshow(s_im{class});
end
end


% -----------------------------------

% Copied functions from problem instruction
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end
end