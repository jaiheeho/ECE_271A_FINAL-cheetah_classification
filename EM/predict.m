function A = predict(x_dcts, dim, mean_BG, sigma_BG, pi_BG, mean_FG, sigma_FG, pi_FG, prior_BG, prior_FG)

%% cheetah Init
Cheetah = imread('cheetah.bmp');
[sizeX, sizeY]=size(Cheetah);
A = rand(sizeX, sizeY);

%% prior cal
for idx_x = 1 : sizeX 
    for idx_y = 1 : sizeY
        x = x_dcts(sizeY*(idx_x-1) + idx_y,:);
        %% EM likelihood
        % BG
        temp_likelyhood_BG = ...
            mixturepdf(x,dim,...
            mean_BG, sigma_BG,pi_BG);
        % FG
        temp_likelyhood_FG = ...
            mixturepdf(x,dim,...
            mean_FG, sigma_FG,pi_FG);
        %% predict
        if prior_BG * temp_likelyhood_BG > prior_FG * temp_likelyhood_FG
            A(idx_x, idx_y) = uint8(0);
        else
            A(idx_x, idx_y) = uint8(1);
        end
    end
end

end
