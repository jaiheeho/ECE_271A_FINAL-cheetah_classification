fprintf('Loading data .. \n');
Cheetah = imread('cheetah.bmp');
Cheetah_mask = imread('cheetah_mask.bmp');
ZigZag = importdata('Zig-Zag Pattern.txt');

load('TrainingSamplesDCT_8_new.mat');
fprintf('Loading data .. done \n');

[nrows_BG, ~] = size(TrainsampleDCT_BG);
[nrows_FG, ncols] = size(TrainsampleDCT_FG);

prior_BG = (nrows_BG)/(nrows_BG + nrows_FG)
prior_FG = (nrows_FG)/(nrows_BG + nrows_FG)

Index_BG = zeros(1,ncols);
Index_FG = zeros(1,ncols);

priorestimate = [nrows_BG, nrows_FG]/(nrows_BG + nrows_FG);
figure();
bar(priorestimate);


M_BG = mean(TrainsampleDCT_BG);
M_FG = mean(TrainsampleDCT_FG);
Std_BG = std(TrainsampleDCT_BG);
Std_FG = std(TrainsampleDCT_FG);
Cov_BG = cov(TrainsampleDCT_BG);
Cov_FG = cov(TrainsampleDCT_FG);
% figure;
% for idx = 1:ncols
%     subplot(8,8,idx);
%     x_BG = linspace(-Std_BG(idx)*4+M_BG(idx),+Std_BG(idx)*4+M_BG(idx));
%     x_FG = linspace(-Std_FG(idx)*4+M_FG(idx),+Std_FG(idx)*4+M_FG(idx));
%     norm_BG = normpdf(x_BG,M_BG(idx),Std_BG(idx));
%     norm_FG = normpdf(x_FG,M_FG(idx),Std_FG(idx));
%     plot(x_BG,norm_BG, x_FG, norm_FG);
%     title(['Coefficient: ',num2str(idx)]);
% end
% figure;
% title('best coefficients');
% i = 1;
% for idx = [1,2, 7, 9, 10, 11, 12, 13]
%     subplot(4,2,i);
%     x_BG = linspace(-Std_BG(idx)*4+M_BG(idx),+Std_BG(idx)*4+M_BG(idx));
%     x_FG = linspace(-Std_FG(idx)*4+M_FG(idx),+Std_FG(idx)*4+M_FG(idx));
%     norm_BG = normpdf(x_BG,M_BG(idx),Std_BG(idx));
%     norm_FG = normpdf(x_FG,M_FG(idx),Std_FG(idx));
%     plot(x_BG,norm_BG, x_FG, norm_FG);
%     title(['Coefficient: ',num2str(idx)]);
%     i = i +1;
% end
% 
% figure;
% title('worst coefficients');
% i = 1;
% for idx =[54, 55, 58, 59, 60, 62, 63, 64]
%     subplot(4,2,i);
%     x_BG = linspace(-Std_BG(idx)*4+M_BG(idx),+Std_BG(idx)*4+M_BG(idx));
%     x_FG = linspace(-Std_FG(idx)*4+M_FG(idx),+Std_FG(idx)*4+M_FG(idx));
%     norm_BG = normpdf(x_BG,M_BG(idx),Std_BG(idx));
%     norm_FG = normpdf(x_FG,M_FG(idx),Std_FG(idx));
%     plot(x_BG,norm_BG, x_FG, norm_FG);
%     title(['Coefficient: ',num2str(idx)]);
%     i = i +1;
% end


[sizeX, sizeY]=size(Cheetah);
Cheetah_norm = double(Cheetah)./255.0;
Cheetah_Padding = zeros(sizeX + 7, sizeY + 7);
Cheetah_Padding(5:sizeX+4, 5:sizeY+4) = Cheetah_norm(:,:);
DCT_block = zeros(8:8);
block = zeros(8:8);
A = zeros(sizeX, sizeY);

x_dcts_all = zeros(1,64);
x_dcts_8   = zeros(1,8);

Cov_BG_8 = cov(TrainsampleDCT_BG(:,[1, 11, 14, 23, 25, 27, 32, 40]));
Cov_FG_8 = cov(TrainsampleDCT_FG(:,[1, 11, 14, 23, 25, 27, 32, 40]));

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
        temp_likelyhood_BG = 1.0;
        temp_likelyhood_FG = 1.0;
%         % 8 dim multi-gaussian 
        i=1;
        for idx= [1, 11, 14, 23, 25, 27, 32, 40]
            [rows, cols] =  find(ZigZag == idx-1);
            temp = DCT_block(rows,cols);
            x_dcts_8(i) = temp;
            i = i+1;
        end
       
        temp_likelyhood_BG = temp_likelyhood_BG * mvnpdf(x_dcts_8,M_BG([1, 11, 14, 23, 25, 27, 32, 40]),Cov_BG_8);
        temp_likelyhood_FG = temp_likelyhood_FG * mvnpdf(x_dcts_8,M_FG([1, 11, 14, 23, 25, 27, 32, 40]),Cov_FG_8);
%         % 64 dim multi-gaussian 
%         for idx_1 = 1 : 8
%             for idx_2 = 1: 8
%                 idx = ZigZag(idx_1,idx_2)+1;
%                 x_dcts_all(idx) = DCT_block(idx_1,idx_2);
%             end
%         end
%         temp_likelyhood_BG = temp_likelyhood_BG * mvnpdf(x_dcts_all,M_BG,Cov_BG);
%         temp_likelyhood_FG = temp_likelyhood_FG * mvnpdf(x_dcts_all,M_FG,Cov_FG);
        if prior_BG * temp_likelyhood_BG > prior_FG * temp_likelyhood_FG
            A(idx_x, idx_y) = uint8(0);
        else
            A(idx_x, idx_y) = uint8(1);
        end
                
        
    end
end

figure;
imagesc(A);
colormap(gray(255));
Correct = 0;


FG_wrong = 0;
BG_wrong = 0;
for idx_x = 1: sizeX
    for idx_y = 1: sizeY
        if A(idx_x, idx_y) == Cheetah_mask(idx_x, idx_y)/255
            Correct = Correct + 1;
        else 
            if A(idx_x, idx_y) == 0
                FG_wrong = FG_wrong +1 ; 
            else
                BG_wrong = BG_wrong +1 ;
             
            end         
        end     
    end
end
cheetahpixels = sum(sum(Cheetah_mask))/255;

prior_cheetah_mask  = cheetahpixels/(sizeX *sizeY)
Error_rate = 1- Correct/(sizeX *sizeY)


FG_wrong_rate = FG_wrong /cheetahpixels;
BG_wrong_rate = BG_wrong /(sizeX * sizeY - cheetahpixels);

Probability_Error = prior_FG * FG_wrong_rate + prior_BG * BG_wrong_rate
