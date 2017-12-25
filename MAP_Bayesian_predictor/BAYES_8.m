function POES_BAYES = BAYES_8(D_BG, D_FG, strategy, alpha)
    D_BG = D_BG(:,[1, 11, 14, 23, 25, 27, 32, 40]);
    D_FG = D_FG(:,[1, 11, 14, 23, 25, 27, 32, 40]);
    Sample_Cov_BG = cov(D_BG);
    Sample_Cov_FG = cov(D_FG);
    Sample_mean_BG = mean(D_BG);
    Sample_mean_FG = mean(D_FG);
    nrows_BG = size(D_BG,1);
    nrows_FG = size(D_FG,1);
    if (strategy == 1)
        load('Prior_1.mat');
    elseif(strategy ==2)
        load('Prior_2.mat');
    end
    mu0_BG = mu0_BG(:,[1, 11, 14, 23, 25, 27, 32, 40]);
    mu0_FG = mu0_FG(:,[1, 11, 14, 23, 25, 27, 32, 40]);
    Cheetah = imread('cheetah.bmp');
    Cheetah_mask = imread('cheetah_mask.bmp');
    ZigZag = importdata('Zig-Zag Pattern.txt');
    
    figure;
    i =1;
    POES_BAYES = zeros(1, size(alpha,2));
    for idx_alpha = 1 : size(alpha,2)
        Sigma_0 = zeros(8,8);
        idx_sigma_0 =1;
        for idx_8 = [1, 11, 14, 23, 25, 27, 32, 40]
            Sigma_0(idx_sigma_0,idx_sigma_0) = W0(idx_8) * alpha(idx_alpha);
            idx_sigma_0 = idx_sigma_0 + 1;
        end
        %% posterior probability p(mu|D1) = G(mu, mu1, Sigma1)
        inv_sigma_BG = inv(Sigma_0 + Sample_Cov_BG./nrows_BG);
        inv_sigma_FG = inv(Sigma_0 + Sample_Cov_FG./nrows_FG);
        Sigma_BG = Sigma_0 * inv_sigma_BG * (Sample_Cov_BG./nrows_BG);
        Sigma_FG = Sigma_0 * inv_sigma_FG * (Sample_Cov_FG./nrows_FG);
        mu_BG = (Sigma_0 * inv_sigma_BG * Sample_mean_BG.' ...
            +  Sample_Cov_BG./nrows_BG * inv_sigma_BG * mu0_BG.').';

        mu_FG = (Sigma_0 * inv_sigma_FG * Sample_mean_FG.' ...
            +  Sample_Cov_FG./nrows_FG * inv_sigma_FG * mu0_FG.').';

        %% BAYES_predicting
        [sizeX, sizeY]=size(Cheetah);
        Cheetah_norm = double(Cheetah)./255.0;
        A = rand(sizeX, sizeY);
        x_dcts_8 = zeros(1,8);
        %% prior cal
        prior_BG = nrows_BG /(nrows_BG + nrows_FG);
        prior_FG = 1- prior_BG;
        for idx_x = 1 : sizeX 
            for idx_y = 1 : sizeY
                x = idx_x - 4;
                y = idx_y - 4;
                if (x < 1) 
                    x = 1;
                end
                if (y < 1)
                    y = 1;
                end
                if (x+7 > sizeX)
                    x = sizeX-7;
                end
                if (y+7 > sizeY)
                    y = sizeY-7;
                end
                block = Cheetah_norm(x:x+7, y:y+7);
                DCT_block = dct2(block);
                idx_8dct=1;
                for idx= [1, 11, 14, 23, 25, 27, 32, 40]
                    [rows, cols] =  find(ZigZag == idx-1);
                    temp = DCT_block(rows,cols);
                    x_dcts_8(idx_8dct) = temp;
                    idx_8dct = idx_8dct+1;
                end
                %% calculate likelihood for BAYESIAN
                temp_likelyhood_BG_BAYES = mvnpdf(x_dcts_8,mu_BG, Sample_Cov_BG + Sigma_BG);
                temp_likelyhood_FG_BAYES = mvnpdf(x_dcts_8,mu_FG, Sample_Cov_FG + Sigma_FG);
                %% predict
                if prior_BG * temp_likelyhood_BG_BAYES > prior_FG * temp_likelyhood_FG_BAYES
                    A(idx_x, idx_y) = uint8(0);
                else
                    A(idx_x, idx_y) = uint8(1);
                end
            end
        end
        %% subplot cheetah
        subplot(3,3,i);
        i =i+1;
        imagesc(A);
        colormap(gray(255));
        title(['BAYESIAN, alpha = ', num2str(alpha(idx_alpha))]);
        %% check correctness
        Correct = 0;
        FG_wrong = 0;
        BG_wrong = 0;
        for idx_x = 1: sizeX
            for idx_y = 1: sizeY
                if (A(idx_x, idx_y) == Cheetah_mask(idx_x, idx_y)/255)
                    Correct = Correct + 1;
                else 
                    if (A(idx_x, idx_y) == 0)
                        FG_wrong = FG_wrong +1 ; 
                    else
                        BG_wrong = BG_wrong +1 ;

                    end         
                end     
            end
        end
        cheetahpixels = sum(sum(Cheetah_mask))/255;
        Error_rate = 1- Correct/(sizeX *sizeY)

        FG_wrong_rate = FG_wrong /cheetahpixels;
        BG_wrong_rate = BG_wrong /(sizeX * sizeY - cheetahpixels);

        Probability_Error = prior_FG * FG_wrong_rate + prior_BG * BG_wrong_rate
        POES_BAYES(idx_alpha) =  Probability_Error;
    end
    title(['Cheetas BAYESIAN', num2str(strategy)]);
end