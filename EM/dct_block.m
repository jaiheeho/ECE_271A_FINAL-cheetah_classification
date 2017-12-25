function x_dcts = dct_block()

Cheetah = imread('cheetah.bmp');
ZigZag = importdata('Zig-Zag Pattern.txt');    
[sizeX, sizeY]=size(Cheetah);
Cheetah_norm = double(Cheetah)./255.0;
dim = 64;
x_dcts = zeros(sizeX*sizeY,dim);
%% prior cal
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
        for idx= 1:dim
            [rows, cols] =  find(ZigZag == idx-1);
            temp = DCT_block(rows,cols);
            x_dcts(sizeY*(idx_x-1) + idx_y,idx) = temp;
        end
    end
end
end
