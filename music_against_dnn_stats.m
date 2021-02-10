%%Statistics used in the paper. IMPORTANT: NOTE that these may slightly 
%differ from the reported figures, depending on how the net converged in
%your case.

clear all, close all

angs_deg = (-30:0.1:30);% +90;
r_m = 0.5:0.1:5.5;
angs_rad = angs_deg*pi/180;
first=1;
load('covsn_predicts.mat');
R_te = permute(R_te,[2 3 1]); %we need to swap dimensions again..
fsize = 14;

%here we limit the range (in meters):
limitrangeidx = y_te(:,2)<=8;
y_te=y_te(limitrangeidx,:);
y_pred=y_pred(limitrangeidx,:);
R_te = R_te(:,:,limitrangeidx);

chooseidx = (1:length(R_te)); %inspect all;

dualmaxcnt=0;
for nn=chooseidx %length(filelist)
    
    P = music2dspectrum(R_te(:,:,nn),angs_rad,r_m);
    theta_k = y_te(nn,1);
    r_k = y_te(nn,2);
    
    theta_hat = y_pred(nn,1);
    r_hat = y_pred(nn,2);
 
    maxv = max(max(P));
    [ri,ci] = find(P==maxv);
    if length(ci)>1
        dualmaxcnt=dualmaxcnt+1; %just to make sure if it peaks in two locs
    end
    ri = ri(1); ci = ci(1);
    theta_hat = angs_rad(ri);
    r_hat = r_m(ci);
    %fprintf('%5.2f : %5.2f\r\n',r_hat,theta);
    err_veca(nn) = theta_hat - theta_k;
    err_vecr(nn) = r_hat - r_k;
    thetahat_vec(nn,1) = theta_hat;
    rhat_vec(nn,1) = r_hat;
    
end

%%
%switch units to degree and lambdas
lambda = 300/1240; %c and 1.240 GHz

MSEa1 = sqrt(sum(((y_pred(:,1)-y_te(:,1))*180/pi).^2)/length(y_te));
MSEr1 = sqrt(sum(((y_pred(:,2)-y_te(:,2))/lambda).^2)/length(y_te));
MAEa1 = sum(abs((y_pred(:,1)-y_te(:,1))*180/pi))/length(y_te);
MAEr1 = sum(abs((y_pred(:,2)-y_te(:,2))/lambda))/length(y_te);

MSDa1 = sum((y_pred(:,1)-y_te(:,1))*180/pi)/length(y_te);
MSDr1 = sum((y_pred(:,2)-y_te(:,2))/lambda)/length(y_te);

MSEa2 = sqrt(sum(((thetahat_vec-y_te(:,1))*180/pi).^2)/length(y_te));
MSEr2 = sqrt(sum(((rhat_vec-y_te(:,2))/lambda).^2)/length(y_te));
MAEa2 = sum(abs((thetahat_vec-y_te(:,1))*180/pi))/length(y_te);
MAEr2 = sum(abs((rhat_vec-y_te(:,2))/lambda))/length(y_te);

MSDa2 = sum((thetahat_vec-y_te(:,1))*180/pi)/length(y_te);
MSDr2 = sum((rhat_vec-y_te(:,2))/lambda)/length(y_te);

fprintf("NFLOPNET:: RMSEangle: %2.2f | RMSEdist %2.2f\n",MSEa1,MSEr1);
fprintf("MUSIC:: RMSEangle: %2.2f | RMSEdist %2.2f\n",MSEa2,MSEr2);

fprintf("NFLOPNET:: MAEangle: %2.2f | MAEdist %2.2f\n",MAEa1,MAEr1);
fprintf("MUSIC:: MAEangle: %2.2f | MAEdist %2.2f\n",MAEa2,MAEr2);

for idx_phi = 1:length(angs_rad)
    for idx_r = 1:length(r_m)
        a = svec(angs_rad(idx_phi),r_m(idx_r));
        
    end
end

function [P] = music2dspectrum(X,phi_vec,r_vec)
    K = 1;
    [U,~,~] = svd(X);
    Un = U(:,(K+1):end); % Noise subspace eigenvectors
    P = zeros(length(phi_vec),length(r_vec));
    for r=1:length(phi_vec)
       for c=1:length(r_vec)
           a = svec(phi_vec(r),r_vec(c));
           P(r,c) = (vecnorm(a)./vecnorm(Un'*a,2,1)).^2;
       end
    end
end

function a = svec(phi,r)
    M=9;
    l=1;
    r=r*1240/300;
    d=0.5;
    m = (0:(M-1)) - 4;    
    a = exp(1j*(-2*pi*d*sin(phi)/l*m + pi*d^2*cos(phi)^2/r*m.^2));
    a = a.';
end
