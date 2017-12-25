function prob = mixturepdf(x_dct, dim, mean, sigma,pi)
% initialize value
class = size(mean,2)/64;
mean_c = zeros(class, dim);
sigma_c = zeros(class*dim, class*dim);
x_dct = x_dct(1:dim); 

% Assignme mean_class , sigma_class,
for idx = 1: class
    mean_c(idx,:) = mean((idx-1)*64 +1 : (idx-1)*64 +dim);
    sigma_c((idx-1)*dim +1:(idx)*dim, (idx-1)*dim +1:(idx)*dim) = ...
        diag(sigma((idx-1)*64+1: (idx-1)*64 +dim));

end
prob = 0.0;
for idx = 1:class  
    prob = prob + (mvnpdf(x_dct,...
        mean_c(idx,:),...
        sigma_c((idx-1)*dim +1:(idx)*dim, (idx-1)*dim +1:(idx)*dim)))...
        *pi(idx);
end
