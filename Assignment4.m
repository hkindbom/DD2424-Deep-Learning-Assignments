% DD2424 Deep Learning - Assignment 4 - Hannes Kindbom

clear all
clc
format long

book_fname = 'Datasets/goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);
book_chars = unique(book_data);

% constants
K = length(book_chars);

[char_to_ind, ind_to_char] = InitCharMappings(book_chars, K);


% Hyperparameters
h = 0.0001;
rng(500)
m = 100;
seq_length = 25;
sig = 0.01;
start_idx = 50;
h_0 = zeros(m,1);
epochs = 10;
nr_upd_steps = floor(epochs*length(book_data)/seq_length);

RNN = InitParameters(sig, m, K);

% Creating word vectors
X_chars_debug = book_data(1:seq_length);
Y_chars_debug = book_data(2:seq_length+1);
X_vecs_debug = CharsToOneHots(X_chars_debug, K, char_to_ind);
Y_vecs_debug = CharsToOneHots(Y_chars_debug, K, char_to_ind);

X_chars = book_data(1:end);
Y_chars = book_data(1:end);
X_vecs = CharsToOneHots(X_chars, K, char_to_ind);
Y_vecs = CharsToOneHots(Y_chars, K, char_to_ind);

% Check Gradients
ComputeGradError(RNN, X_vecs_debug, Y_vecs_debug, h_0, h)

% Train RNN
disp('Training...');
RNNstar = TrainRNN(RNN, X_vecs, Y_vecs, h_0, seq_length, nr_upd_steps, ind_to_char);

% Generate text
x_0 = CharIdxToOneHot(start_idx, K);
GenerateText(RNNstar, h_0, x_0, 1000, ind_to_char)

% Function definitions
function RNNstar = TrainRNN(RNN, X_vecs, Y_vecs, h_0, seq_length, nr_upd_steps, ind_to_char)
max_idx = size(X_vecs, 2) - seq_length;
m = InitM(RNN);
e = 1;
smooth_loss_all = zeros(nr_upd_steps, 1);

for upd_step = 1:nr_upd_steps
    X_vecs_seq = X_vecs(:, e:e+seq_length-1);
    Y_vecs_seq = Y_vecs(:, e+1:e+seq_length);
    
    if e == 1
        h_t = h_0;
    end
    
    if mod(upd_step, 10000) == 0
        disp(upd_step);
        disp(smooth_loss);
        GenerateText(RNN, h_t, X_vecs_seq(:,1), 1000, ind_to_char);
    end
    
    [Grads, h_t, loss] = ComputeGradients(RNN, X_vecs_seq, Y_vecs_seq, h_t);
    
    if upd_step == 1
        smooth_loss = loss;
        disp(upd_step);
        disp(smooth_loss);
        GenerateText(RNN, h_0, X_vecs_seq(:,1), 200, ind_to_char);
    end
    smooth_loss = 0.999*smooth_loss + 0.001 * loss;
    smooth_loss_all(upd_step) = smooth_loss;
    
    [RNN, m] = AdaGrad(Grads, RNN, m);    
    e = GetNewSeqStart(e, max_idx, seq_length);    
end
RNNstar = RNN;

upd_steps = linspace(1, nr_upd_steps, nr_upd_steps);
plot(upd_steps, smooth_loss_all);
title('Smooth loss Progress')
xlabel('Upd Steps')
ylabel('Smooth loss')
end

function GenerateText(RNN, h_0, x_0, seq_length, ind_to_char)
Y = SynthesizeText(RNN, h_0, x_0, seq_length);
text = OneHotsToText(Y, ind_to_char);
disp(text);
end

function e = GetNewSeqStart(e, max_idx, seq_length)
e = e + seq_length;
if e > max_idx
    e = 1;
end
end

function m = InitM(RNN)
for f = fieldnames(RNN)'
    m.(f{1}) = zeros(size(RNN.(f{1})));
end
end

function [RNN, m] = AdaGrad(Grads, RNN, m)
eta = 0.1;
eps = 1e-8;
for f = fieldnames(RNN)'
    m.(f{1}) = m.(f{1}) + Grads.(f{1}).^2;
    RNN.(f{1}) = RNN.(f{1}) - eta*(Grads.(f{1}) ./ (m.(f{1}) + eps).^(0.5));
end
end

function one_hots = CharsToOneHots(chars, K, char_to_ind)
one_hots = zeros(K, length(chars));
for char_i = 1:length(chars)
    one_hots(:, char_i) = CharIdxToOneHot(char_to_ind(chars(char_i)), K);
end
end

function text = OneHotsToText(one_hots, ind_to_char)
text = zeros(1, size(one_hots,2));
for t = 1:size(one_hots,2)
    char_idx = find(one_hots(:, t));
    text(t) = ind_to_char(char_idx);
end
text = char(text);
end

function ComputeGradError(RNN, X_vecs, Y_vecs, h_0, h)
[Grads_an, ~, ~] = ComputeGradients(RNN, X_vecs, Y_vecs, h_0);
Grads_num = ComputeGradsNum(X_vecs, Y_vecs, RNN, h);

rel_grad_error_W = ComputeRelativeError(Grads_an.W, Grads_num.W)
rel_grad_error_V = ComputeRelativeError(Grads_an.V, Grads_num.V)
rel_grad_error_U = ComputeRelativeError(Grads_an.U, Grads_num.U)
rel_grad_error_b = ComputeRelativeError(Grads_an.b, Grads_num.b)
rel_grad_error_c = ComputeRelativeError(Grads_an.c, Grads_num.c)
end

function relative_error = ComputeRelativeError(grad_an, grad_num)
eps = 0.0001;
relative_error = norm(grad_an - grad_num) / max([eps, norm(grad_an) + norm(grad_num)]);
end

function [Grads, h_t, loss] = ComputeGradients(RNN, X_vecs, Y_vecs, h_0)
n = size(X_vecs, 2);

[loss, H, P] = ForwardPass(RNN, X_vecs, Y_vecs, h_0);
Grads.V = zeros(size(RNN.V));
Grads.W = zeros(size(RNN.W));
Grads.U = zeros(size(RNN.U));
Grads.b = zeros(size(RNN.b));
Grads.c = zeros(size(RNN.c));


for t = n:-1:1
    x_t = X_vecs(:, t);
    y_t = Y_vecs(:, t);
    p_t = P(:, t);
    h_t = H(:, t);

    dL_do_t = -transpose(y_t - p_t);
    
    Grads.c = Grads.c + transpose(dL_do_t);    
    Grads.V = Grads.V + transpose(dL_do_t)*transpose(h_t);
    
    if t == n
        dL_dh_t = dL_do_t * RNN.V;
    else
        dL_dh_t = dL_do_t*RNN.V + dL_da_t * RNN.W;
    end
    Grads.b = Grads.b + diag(1 - h_t.^2)*transpose(dL_dh_t);
    
    a_t = atanh(h_t);
    dL_da_t = dL_dh_t * diag(1 - tanh(a_t).^2);
    if t == 1
        h_t_min_1 = h_0;
    else
        h_t_min_1 = H(:, t-1);
    end
    Grads.W = Grads.W + transpose(dL_da_t) * transpose(h_t_min_1);
    Grads.U = Grads.U + transpose(dL_da_t) * transpose(x_t);
end

Grads = ClipGradients(Grads);
h_t = H(:, n);
end

function Grads = ClipGradients(Grads)
for f = fieldnames(Grads)'
    Grads.(f{1}) = max(min(Grads.(f{1}), 5), -5);
end
end

function [loss, outputs, P] = ForwardPass(RNN, X_vecs, Y_vecs, h_0)
loss = 0;
outputs = zeros(size(h_0, 1), size(X_vecs, 2));
P = zeros(size(X_vecs));
h_t = h_0;
for t = 1:size(X_vecs, 2)
    x_t = X_vecs(:, t);
    y_t = Y_vecs(:, t);

    [p_t, h_t] = SingleForwardPass(RNN, h_t, x_t);
    loss = loss - log(transpose(y_t) * p_t);
    outputs(:, t) = h_t;
    P(:, t) = p_t;
end
end

function loss = ComputeLoss(X_vecs, Y_vecs, RNN, h_0)
[loss, ~, ~] = ForwardPass(RNN, X_vecs, Y_vecs, h_0);
end

function [p, h_t_next] = SingleForwardPass(RNN, h_t, x_t)
a_t = RNN.W*h_t + RNN.U*x_t + RNN.b;
h_t_next = tanh(a_t);
o_t = RNN.V*h_t_next + RNN.c;
p = SoftMax(o_t);
end

function Y = SynthesizeText(RNN, h_0, x_0, n)
K = size(RNN.c, 1);
h_t = h_0;
x_t = x_0;
Y = zeros(K, n);

for t = 1:n
    [p_t, h_t] = SingleForwardPass(RNN, h_t, x_t);
    
    next_char_idx = SampleNextX(p_t);    
    x_t = CharIdxToOneHot(next_char_idx, K);
    Y(:, t) = x_t;
end
end

function one_hot_char = CharIdxToOneHot(next_char_idx, K)
one_hot_char = zeros(K, 1);
one_hot_char(next_char_idx, 1) = 1;
end

function char_idx = SampleNextX(p)
cp = cumsum(p);
a = rand;
ixs = find(cp-a >0);
char_idx = ixs(1);
end

function P = SoftMax(s)
P = exp(s)./(transpose(ones(size(s)))*exp(s));
end

function [char_to_ind, ind_to_char] = InitCharMappings(book_chars, K)
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');
for char_idx = 1:K
    char_to_ind(book_chars(char_idx)) = char_idx;
    ind_to_char(char_idx) = book_chars(char_idx);
end
end

function RNN = InitParameters(sig, m, K)
RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;
end


% Functions copied from problem instructions
function num_grads = ComputeGradsNum(X, Y, RNN, h)

for f = fieldnames(RNN)'
    disp('Computing numerical gradient for')
    disp(['Field name: ' f{1} ]);
    num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
end
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
hprev = zeros(size(RNN.W, 1), 1);
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
    l1 = ComputeLoss(X, Y, RNN_try, hprev);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    l2 = ComputeLoss(X, Y, RNN_try, hprev);
    grad(i) = (l2-l1)/(2*h);
end
end