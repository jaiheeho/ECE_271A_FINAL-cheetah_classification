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

classes = [1 2 4 8 16 32];
dims = [1 2 4 8 16 24 32 40 48 56 64];

poes = zeros(size(classes,2),size(dims,2));

%% train and test
for idx_c  = 1:size(classes,2)
    %% parameters
    class = classes(idx_c);
    [mean_c_BG, sigma_c_BG,pi_c_BG] =  EM(64, class, TrainsampleDCT_BG);
    [mean_c_FG, sigma_c_FG,pi_c_FG] =  EM(64, class, TrainsampleDCT_FG);
    mean_BG = mean_c_BG;
    mean_FG = mean_c_FG;
    sigma_BG = sigma_c_BG;
    sigma_FG= sigma_c_FG;
    pi_BG = pi_c_BG;
    pi_FG = pi_c_FG; 
    for idx_dim = 1:size(dims,2)
        dim = dims(idx_dim);
        %% EM for BG
        A =  predict(x_dcts, dim,...
            mean_BG, sigma_BG, pi_BG,...
            mean_FG, sigma_FG, pi_FG,...
            prior_BG,prior_FG);
        poes(idx_c,idx_dim) = evaluate(A,prior_BG,prior_FG);
        imageName = 'Cheetah C=%d(dim: %d, poe:%1.4f)';
        name = sprintf(imageName,class,dim,poes(idx_c,idx_dim));
        fig =  figure(); figure(fig);imagesc(A); colormap(gray(255));
        title(name);
        filename = './cheetah_result/C_%d(dim_%d).png';
        filename = sprintf(filename,class,dim);
        saveas(fig,filename);
    end
end


%% plot poe vs dimension
colors = ['r','g','b','c','m','y'];

fig = figure(); figure(fig);
hold on
for idx = 1:size(classes,2)
    %% plot
    plot(dims,poes(idx,:),colors(rem(idx,size(colors,2))+1));
end
hold off
legend({'C:1', 'C:2','C:4','C:8','C:16','C:32'});

title('poe vs dim with class = [1,2,4,...]');
filename = './plot/poe_Class.png';
filename = sprintf(filename);
saveas(fig,filename); 


%%
fig = figure(); figure(fig);
semilogx(classes,poes(:,7),'r');
title('poe vs Class[1,2,4,...] (dim:32)');
filename = './plot/poe_dim(32).png';
filename = sprintf(filename);
saveas(fig,filename); 
