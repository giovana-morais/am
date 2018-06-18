function out = poly_features(X)
 % generate new features with different polynomial degree
	grau = 3;
	out = ones(size(X(:,1)));  
  
	for i = 1:grau
		for j = 0:i
		    out(:, end+1) = (X(:,1).^(i-j)).*(X(:,4).^j);
		end
	end
end