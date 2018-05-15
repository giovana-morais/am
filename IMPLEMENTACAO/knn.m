function ypred = knn(train, train_labels, test, K) 
% Implementacao do knn para o problema de reconhecimento de atividades humanas

% como ja temos os dados normalizados, devemos primeiramente calcular as distancias euclidianas
% entre os elementos da base de treinamento com os elementos da base de dados

ind_viz = zeros(K,1);  
% vec_labels = zeros(K,1);
D = zeros(length(train),1);

for i = 1:length(train)
	D(i) = norm((train(i,:) - test(1,:)), 2);
endfor 

% pega os K menores objetos do vetor de distancias
%ind_viz = find(D == min(D), K)
[temp, idx] = sort(D);
ind_viz = train_labels(idx(1:K));

ypred = mode(ind_viz);

% =========================================================================

end