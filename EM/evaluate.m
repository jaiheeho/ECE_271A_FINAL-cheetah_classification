function poe = evaluate(A,prior_BG,prior_FG)

Cheetah_mask = imread('cheetah_mask.bmp');
[sizeX, sizeY]=size(Cheetah_mask);
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
Error_rate = 1- Correct/(sizeX *sizeY);
FG_wrong_rate = FG_wrong /cheetahpixels;
BG_wrong_rate = BG_wrong /(sizeX * sizeY - cheetahpixels);

poe = prior_FG * FG_wrong_rate + prior_BG * BG_wrong_rate

end