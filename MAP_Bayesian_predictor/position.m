function [x,y] = position(idx_x, idx_y, sizeX, sizeY)
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
end