%Process measured data for the DNN-DOA. This script precomputes the 
%covariance estimates, since doing this in python was way too slow
%
%The output is written into covariances folder, where each .mat
%file contains the vectorized covariances and true locations of transmitter
%The data has already been centered to mean, i.e DC-offsets removed.
%mikko.t.laakso@aalto.fi
clear all;

%num snapshots for each covariance mtx
snapshots = 256; 
%mask for vectorising the cov upper tri
mask = triu(true(size(ones(9,9))),1); 

%create the directory
mkdir covariances

for k=1:143
    load(['signals/signal' int2str(k) '.mat']);
    istart=1;
    nc = 1;
    r_v=[];
    RM = [];
    %vectorize N/snapshots covariances from the signal matrix
    while istart+snapshots-1 <= size(X,1)
        %Covariance
        R = X(istart:istart+snapshots-1,:)'*X(istart:istart+snapshots-1,:)/snapshots;
        
        %save the full matrix form
        RM(:,:,nc) = R;
        
        %vectorize the upper triangle of the cov estimate:
        R = R(mask);
        R = [real(R.') imag(R.')];
        r_v(nc,:) = R;
        
        %make suitable length vectors corresponding to the true locs
        theta_k(nc,1) = theta_rad;
        r_k(nc,1) = r_m;
        nc = nc + 1;
        istart = istart + snapshots;
    end
    save(['covariances/cov' num2str(k) '.mat'],'r_v','RM','r_k','theta_k','snapshots');
    k=k+1;
end
