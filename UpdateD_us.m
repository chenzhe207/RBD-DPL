function [TempD,Iter] = UpdateD( train_data, A, D )
rho = 1;
rate_rho = 1.2;
Imat= eye(size(A,1));
previousD = D;
TempS = D;
TempT  = zeros(size(TempS));
Iter = 1;ERROR=1;
while(ERROR>1e-6&&Iter<100)
     TempD   = (rho*TempS-TempT+train_data*A')/(rho*Imat+A*A');
     TempS   = normcol_lessequal(TempD+rho\TempT);
     TempT   = TempT+rho*(TempD-TempS);
     rho     = rate_rho*rho;
     ERROR = mean(mean((previousD- TempD).^2));
     previousD = TempD;
     Iter=Iter+1;
end     
end
