function [mean, sigma,pi_c] = EM(dim, class, DCT)
%% Given information
[nrows, ~] = size(DCT);
d = DCT(:,1:dim);
%% Random assignment of class parameters
mean_c = -3 + (6)*rand(class,dim);
sigma_c = 1*rand(dim*class, dim*class)+3;
sigma_c = diag(diag(sigma_c));
pi_c = randi([10 50],1, class);
pi_c = pi_c/sum(pi_c);
z = zeros(nrows, class);

epsilon = zeros(size(sigma_c));
epsilon(:,:) = 0.0001;
%% EM
for iteration = 1:100
   %% E
    for idx = 1:nrows
        %% iter
        x_likelihood = zeros(1, class);
        for idx_class = 1:class
            temp_sigma = sigma_c((idx_class-1)*(dim)+1:idx_class*dim,(idx_class-1)*(dim)+1:idx_class*dim);
            temp_mean = mean_c(idx_class,:);
            x_likelihood(idx_class) = mvnpdf(d(idx,:),temp_mean, temp_sigma) *pi_c(idx_class);
        end 
        z(idx,:) = x_likelihood/sum(x_likelihood);
    end
    pi_c = sum(z)/nrows
    
    %% M  
    for idx_class = 1: class
        %% iteration
        temp = (d - repmat(mean_c(idx_class,:),nrows,1))...
            .* (repmat(z(:, idx_class),1,dim));
%         fprintf('sum class %d, %4.4f  ', idx_class, sum(z(:, idx_class)));
        sigma_c((idx_class-1)*(dim)+1:idx_class*dim,(idx_class-1)*(dim)+1:idx_class*dim) = ...
            transpose(temp) * (d - repmat(mean_c(idx_class,:),nrows,1)) /sum(z(:, idx_class));
        mean_c(idx_class,:) = sum(d .* repmat(z(:, idx_class),1,dim))/sum(z(:, idx_class));
    end 
    sigma_c = sigma_c + epsilon;
    sigma_c = diag(diag(sigma_c));
end

mean = zeros(1,dim*class);
for idx_c = 1:class
    mean((idx_c-1)*dim +1: (idx_c)*dim) = mean_c(idx_c,:);
end
sigma = diag(sigma_c).';
end