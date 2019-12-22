function [ Dinit, Pinit, Ainit, Minit, Winit, Q, L ] = Initilization( train_data, train_label, Htr, DictSize, c, train_num )


[Dinit] = initializationDictionary(train_data, Htr, DictSize, 10, 30); %initialize D
Ainit = (Dinit' * Dinit + 1e-4 * eye(DictSize)) \ (Dinit' * train_data); %initialize A
Xtemp = pinv(train_data * train_data'); %precompute inv(XX')
Pinit = Ainit * train_data' * Xtemp; %initialize P
Q = [];
for i  = 1 : c
    temp = ones(DictSize / c, train_num);
    Q = blkdiag(Q, temp); 
end %define Q
Minit = Q; %initialize M

L = Construct_L(train_data', train_label);  

Winit = (Htr * Ainit') / (Ainit * Ainit' + 1e-4 * eye(DictSize));
end

