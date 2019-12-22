clear all
clc
close all
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\OMPbox')); % add sparse coding algorithem OMP

load randomprojection_AR.mat; 
DATA = DATA./ repmat(sqrt(sum(DATA .* DATA)), [size(DATA, 1) 1]); %normalize
c = length(unique(Label));
numClass = zeros(c,1);
for i = 1 : c
    numClass(i, 1) = length(find(Label == i));
end
%% select training and test samples
train_num = 10;
DictSize = 500;
for ii = 1 : 10
fprintf('\nii = %d; ', ii);

train_data = []; test_data = []; 
train_label = []; test_label = [];
for i = 1 : c
    index = find(Label == i); 
    randindex = index(randperm(length(index)));
    train_data = [train_data DATA(:,randindex(1 : train_num))];
    train_label = [train_label  Label(randindex(1 : train_num))];
  
    test_data = [test_data DATA(:, randindex(train_num + 1 : end))];
    test_label = [test_label  Label(randindex(train_num + 1 : end))];
end

for i = 1 : size(train_data, 2)
    a = train_label(i);
    Htr(a, i) = 1;
end
for i = 1 : (size(test_data, 2))
    a = test_label(i);
    Htt(a, i) = 1;
end

%Parameter setting

alpha = 1e-4;
beta = 1e-1;
gamma = 1e-3;
lambda = 1e-3;
% fprintf('\nii = %d; ', ii);
% DPL trainig
tic
[ D, P, A, M, W, value] = TrainDPL(train_data, train_label, train_num, Htr, c, DictSize, alpha, beta, gamma, lambda);

%DPL testing
tic
PredictCoef = P * test_data;
accuracy(ii) = classification(W, Htt, PredictCoef);
test_time(ii) = toc;

%Show accuracy and time
fprintf('; Recognition rate for DPL is : %.03f', accuracy(ii));

% fprintf('; Max Recognition rate for DPL is : %.03f', max(accuracy));
end

mean(accuracy)
std(accuracy)


x=[0,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1];
y=[77.75 77.50 77.75 77.75 77.25 77.25 75.25 74.75 73.00 72.25];
plot(y)
grid on;
