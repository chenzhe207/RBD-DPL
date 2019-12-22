function [ D, P, A, M, W, value] = TrainDPL( X, train_label, train_num, Htr, c, K, alpha, beta, gamma, lambda)

%% Initialize L, D, P and M
[ D, P, A, M, W, Q, L ] = Initilization( X, train_label, Htr, K, c, train_num );
B = 2 * Q - ones(size(Q));
temp = inv(alpha * X * X' + gamma * X * L * X' + 1e-4 * eye(size(X, 1)));
k = K / c;
MaxIter = 10;
%% Alternatively update P, D, M and A
fprintf('Iter =');
for Iter = 1 : MaxIter
    if mod(Iter, 5)==0 || Iter == 1 
      fprintf(' %d', Iter);
    end
    %update A
    A = (D' * D + (alpha + beta) * eye(K) + lambda * W' * W) \ (D' * X + alpha * P * X + beta * (Q .* M) + lambda * W' * Htr);
    %update P
    P = alpha * A * X' * temp;  
    %update M   
%     M = A + (ones(size(Q)) - Q) .* M;
     M = 2 * A - B .* M;
    %update W
    W = Htr * A' / (A * A' + eye(K)); 
    %update D
    [D, iter] = UpdateD_us(X, A, D);
    value(Iter) = norm(X - D * A, 'fro') ^ 2 + alpha * norm(P * X - A, 'fro') ^ 2 + beta * norm(A - Q .* M, 'fro') ^ 2 + gamma * trace(P * X * L * X' * P') + lambda * (norm(Htr - W * A, 'fro') ^ 2 + norm(W, 'fro') ^ 2);
end
end
    
