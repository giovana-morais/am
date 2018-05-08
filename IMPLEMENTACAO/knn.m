function ypred = knn(train, train_labels, test, K) 
% Implementacao do knn para o problema de reconhecimento de atividades humanas

% como ja temos os dados normalizados, devemos primeiramente calcular as distancias euclidianas
% entre os elementos da base de treinamento com os elementos da base de dados

ind_neigh = zeros(K,1);  
vec_labels = zeros(K,1);
D = zeros(length(train),1);

for i = 1:length(train)
	D(i) = norm((train(i,:) - test(1,:)), 2);
endfor

for aux = 1:K
  dist = intmax('int32');
	for i = 1:length(train)
		if ~ismember(i, ind_neigh) && D(i,1) < dist
		  dist = D(i,1);
		  neigh = i;
		endif 
	endfor
  ind_neigh(aux,1) = neigh;
  vec_labels(aux,1) = train_labels(neigh,1); 
endfor

ypred = mode(vec_labels);

end