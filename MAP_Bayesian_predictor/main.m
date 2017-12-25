clear 
fprintf('Loading data .. \n');
load('TrainingSamplesDCT_subsets_8.mat')
load('Alpha.mat');
fprintf('Loading data .. done \n');

D_BG = D2_BG;
D_FG = D2_FG;

%%
for idx = 1:2
    %% ML
    POES_ML = ML(D_BG, D_FG, idx, alpha)
    %% MAP
    POES_MAP = MAP(D_BG, D_FG, idx, alpha)    
    %% BAYES
    POES_BAYES = BAYES(D_BG, D_FG, idx, alpha)
    %% plot
    figure;
    semilogx(alpha,POES_ML,'r', alpha,POES_MAP,'g',alpha,POES_BAYES,'b');
    legend('ML','MAP','BAYES')
    title(['ML, MAP, BAYES PoE vs Alpha ', 'Stratege: ', num2str(idx)])
end




