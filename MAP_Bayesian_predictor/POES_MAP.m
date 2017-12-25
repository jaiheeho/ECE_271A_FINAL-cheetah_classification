function POES_MAP = ML(D_BG, D_FG, strategy, alpha)
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
    
    load('Prior_1.mat');
    Cheetah = imread('cheetah.bmp');
    Cheetah_mask = imread('cheetah_mask.bmp');
    ZigZag = importdata('Zig-Zag Pattern.txt');
    
    figure;
    title('Cheetas ML');
    i =1;
    POES_MAP = zeros(1, size(alpha,2));
    for idx_alpha = 1 : size(alpha,2)
        Sigma_0 = zeros(64,64);
        for idx = 1:64
            Sigma_0(idx,idx) = W0(idx) * alpha(idx_alpha);
        end
        %posterior probability p(mu|D1) = G(mu, mu1, Sigma1)
        inv_sigma_BG = inv(Sigma_0 + Sample_Cov_BG./nrows_BG);
        inv_sigma_FG = inv(Sigma_0 + Sample_Cov_FG./nrows_FG);

        Sigma_BG = Sigma_0 * inv_sigma_BG * (Sample_Cov_BG./nrows_BG);
        Sigma_FG = Sigma_0 * inv_sigma_FG * (Sample_Cov_BG./nrows_FG);

        mu_BG = (Sigma_0 * inv_sigma_BG * Sample_mean_BG.' ...
            +  Sample_Cov_BG./nrows_BG * inv_sigma_BG * mu0_BG.').';

        mu_FG = (Sigma_0 * inv_sigma_FG * Sample_mean_FG.' ...
            +  Sample_Cov_FG./nrows_FG * inv_sigma_FG * mu0_FG.').';

        %% P1-D1-predicting

        [sizeX, sizeY]=size(Cheetah);
        Cheetah_norm = double(Cheetah)./255.0;
        Cheetah_Padding = zeros(sizeX + 7, sizeY + 7);
        Cheetah_Padding(5:sizeX+4, 5:sizeY+4) = Cheetah_norm(:,:);
        A = zeros(sizeX, sizeY);
        x_dcts_all = zeros(1,64);

        prior_BG = nrows_BG /(nrows_BG + nrows_FG);
        prior_FG = 1- prior_BG;

        for idx_x = 5 : sizeX + 4
            for idx_y = 5 : sizeY + 4
                block = Cheetah_Padding(idx_x-4:idx_x+3, idx_y-4:idx_y+3);
                DCT_block = dct2(block);

                % 64 dim multi-gaussian 
                for idx_1 = 1 : 8
                    for idx_2 = 1: 8
                        idx = ZigZag(idx_1,idx_2)+1;
                        x_dcts_all(idx) = DCT_block(idx_1,idx_2);
                    end
                end
                % calculate likelihood for MAP
    
                temp_likelyhood_BG_MAP = mvnpdf(x_dcts_all,mu_BG, Sample_Cov_BG);
                temp_likelyhood_FG_MAP = mvnpdf(x_dcts_all,mu_FG, Sample_Cov_FG);
                if prior_BG * temp_likelyhood_BG_MAP > prior_FG * temp_likelyhood_FG_MAP
                    A(idx_x-4, idx_y-4) = uint8(0);
                else
                    A(idx_x-4, idx_y-4) = uint8(1);
                end


            end
        end
        subplot(3,3,i);
        i =i+1;
        imagesc(A);
        colormap(gray(255));
        title(['alpha = ', num2str(alpha(idx_alpha))]);
        
%         figure;
%         imagesc(A);
%         colormap(gray(255));
%         title(['ML : alpha = ', num2str(alpha(idx_alpha))]);
        
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
        POES_MAP(idx_alpha) =  Probability_Error;
    end
end