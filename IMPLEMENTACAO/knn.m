function ypred = knn(train, train_labels, test, K) 
% Implementacao do knn para o problema de reconhecimento de atividades humanas

% inicializa vetor de indices dos vizinhos mais proximos
ind_viz = zeros(K,1);
% inicializa vetor de distancias
D = zeros(length(train),1);

% calcula distancia euclidiana entre todos elementos da amostra de treino com a amostra de teste
for i = 1:length(train)
	D(i) = norm((train(i,:) - test(1,:)), 2);
endfor 

% pega os K menores objetos do vetor de distancias
[temp, idx] = sort(D);
ind_viz = train_labels(idx(1:K));

% pega a moda dos indices dos K vizinhos mais proximos para predizer o resultado
ypred = mode(ind_viz);

% =========================================================================

end