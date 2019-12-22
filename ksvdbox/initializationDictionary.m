
function [Dinit]=initializationDictionary(training_feats,H_train,dictsize,iterations,sparsitythres)

numClass = size(H_train,1); % number of objects
numPerClass = round(dictsize/numClass); % initial points from each classes
Dinit = []; % for LC-Ksvd1 and LC-Ksvd2
dictLabel = [];
for classid=1:numClass
    col_ids = find(H_train(classid,:)==1);
    data_ids = find(colnorms_squared_new(training_feats(:,col_ids)) > 1e-6);   % ensure no zero data elements are chosen
    %    perm = randperm(length(data_ids));
    perm = [1:length(data_ids)]; 
    %%%  Initilization for LC-KSVD (perform KSVD in each class)
    Dpart = training_feats(:,col_ids(data_ids(perm(1:numPerClass))));
    para.data = training_feats(:,col_ids(data_ids));
    para.Tdata = sparsitythres;
    para.iternum = iterations;
    para.memusage = 'high';
    % normalization
    para.initdict = normcols(Dpart);
    % ksvd process
    [Dpart,Xpart,Errpart] = ksvd(para,'');
    Dinit = [Dinit Dpart];
    
    labelvector = zeros(numClass,1);
    labelvector(classid) = 1;
    dictLabel = [dictLabel repmat(labelvector,1,numPerClass)];
end

