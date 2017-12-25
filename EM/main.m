clear 
fprintf('Loading data .. \n');
load('TrainingSamplesDCT_8_new.mat');
Cheetah = imread('cheetah.bmp');
Cheetah_mask = imread('cheetah_mask.bmp');
ZigZag = importdata('Zig-Zag Pattern.txt');    
fprintf('Loading data .. done \n');

%% Given information
[nrows_BG, ~] = size(TrainsampleDCT_BG);
[nrows_FG, ~] = size(TrainsampleDCT_FG);
prior_BG = nrows_BG /(nrows_BG + nrows_FG);
prior_FG = 1- prior_BG; 
[sizeX, sizeY]=size(Cheetah);
x_dcts = dct_block();

%% parameters
dims = [1 2 4 8 16 24 32 40 48 56 64];
class = 8;
num_mix = 5;
mean_BG = zeros(num_mix, 64*class);
mean_FG = zeros(num_mix, 64*class);
sigma_BG = zeros(num_mix, 64*class);
sigma_FG = zeros(num_mix, 64*class);
pi_BG = zeros(num_mix, class);
pi_FG = zeros(num_mix, class);

%% training mixtures
for  idx= 1:5
    [mean_c_BG, sigma_c_BG,pi_c_BG] =  EM(64, class, TrainsampleDCT_BG);
    [mean_c_FG, sigma_c_FG,pi_c_FG] =  EM(64, class, TrainsampleDCT_FG);
    mean_BG(idx,:) = mean_c_BG;
    mean_FG(idx,:) = mean_c_FG;
    sigma_BG(idx,:) = sigma_c_BG;
    sigma_FG(idx,:) = sigma_c_FG;
    pi_BG(idx,:) = pi_c_BG;
    pi_FG(idx,:) = pi_c_FG; 
    fprintf('Traingint %d/%d DONE\n', idx,5);
end
%% Probability of error
poes = zeros(25:size(dims,2));
%% predicting 
for idx1= 5:5
    for idx2 = 2:5
        idx = (idx1-1) * 5 + idx2;
        for idx_dim = 1:size(dims,2)
            dim = dims(idx_dim);
            %% EM for BG
            A =  predict_2(x_dcts, dim, idx1, idx2,...
                mean_BG, sigma_BG, pi_BG,...
                mean_FG, sigma_FG, pi_FG,...
                prior_BG,prior_FG);
            poes(idx,idx_dim) = evaluate(A,prior_BG,prior_FG);
            imageName = 'Cheetah %d,%d(dim: %d, poe:%1.4f)';
            name = sprintf(imageName,idx1,idx2,dim,poes(idx,idx_dim));
            fig =  figure(); figure(fig);imagesc(A); colormap(gray(255));
            title(name);
            filename = './cheetah_result/%d_%d(dim_%d).png';
            filename = sprintf(filename,idx1,idx2,dim);
            saveas(fig,filename);
        end
    end
end


%% plot poe vs dimension
for idx = 1:5
    %% plot
    fig = figure(); figure(fig);
    plot(dims,poes((idx-1)*5+1,:),'r',dims,poes((idx-1)*5+2,:),'g',dims,poes((idx-1)*5+3,:),'b',...
        dims,poes((idx-1)*5+4,:),'y',dims,poes((idx-1)*5+5,:),'m');
    legend({'FG_1', 'FG_2','FG_3', 'FG_4', 'FG_5'});
    imageName = 'Cheetah BG : %d and others';
    name = sprintf(imageName,idx);
    title(name);
    filename = './plot/BG_%d.png';
    filename = sprintf(filename,idx);
    saveas(fig,filename);  
end







