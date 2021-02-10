%Near-field localization using machine learning: an empirical study

%This script produces images used in the article - in it's current form
%it makes an animation comparing the ground truth location, music spectrum
%and NFLOPnet estimators (black circle).

%Mikko Laakso
%mikko.t.laakso@aalto.fi

%be sure to run computecovs.m first, then python3 NFLOPnet.py before 
%running this, or else the required resultfiles don't exist.

%Unfortunately the data is now in randomized order, as it comes from 
%the train_test_split function, hence the jumping locations.

clear all, close all

%% parameters
%params for plotting and animation:
first=1;
angs_deg = (-30:0.1:30);
angs_rad = angs_deg*pi/180;
r_m = 0.5:0.1:5.5;

load('covsn_predicts.mat');
%coming from python, we need to swap dimensions again...
R_te = permute(R_te,[2 3 1]);
fsize = 14;

%which samples to include (the gif animation gets bloated if too many):
chooseidx = 1:200;
first = 1; % toggle for the gif animation.
%% animation loop:
for nn=chooseidx
    
    P = music2dspectrum(R_te(:,:,nn),angs_rad,r_m);
    imagesc(angs_deg,r_m,P');
    set(gca,'YDir','normal') ;
    hold on;
    theta_k = y_te(nn,1);
    r_k = y_te(nn,2);
    
    theta_hat = y_pred(nn,1);
    r_hat = y_pred(nn,2);
    
    xlim([-35 35]);
    ylim([0 6]);
   
    plot(theta_k*180/pi,r_k,'rx','MarkerSize',20,'LineWidth',3)
    plot(theta_hat*180/pi,r_hat,'ko', 'MarkerSize', 20,'LineWidth',3);
    xlim([min(angs_deg) max(angs_deg)]);
    ylim([min(r_m) max(r_m)]);
    ax=gca;
    ax.FontSize=fsize;
    xlabel('$\hat{\theta}$','interpreter','Latex','FontSize',fsize+6);
    ylabel('$\hat{r}$','interpreter','Latex','FontSize',fsize+6);
    
    drawnow;
    fh = gcf;
    %pause(1);
    
    %saveas(gcf,['spectrum' int2str(nn) '.png']);
    
    if (first>0)
        gif('anim.gif','DelayTime',1,'LoopCount',Inf,'frame',fh);
        first=0;
    else
        gif
    end

    hold off;
    maxv = max(max(P));
    [ri,ci] = find(P==maxv);
    ri = ri(1); ci = ci(1);
    theta = angs_rad(ri);
    r_hat = r_m(ci);
end

%MUSIC estimator:
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

%steering vectors for our array
function a = svec(phi,r)
    M=9;
    l=1;
    r=r*1240/300;
    d=0.5;
    m = (0:(M-1)) - 4;    
    a = exp(1j*(-2*pi*d*sin(phi)/l*m + pi*d^2*cos(phi)^2/r*m.^2));
    a = a.';
end
