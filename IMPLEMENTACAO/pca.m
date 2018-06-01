function data_PCA = pca(all_data)
  
% base de dados sem a coluna de classes  
data_wc = all_data(:,1:end-1);

printf("Iniciando calculos do PCA...\n\n");
fflush(stdout);
% calcula a matriz de covariancia
coma = cov(data_wc);

% calcula os autovalores e autovetores da matriz de covariancia
[eigvectors, eigvalues, ~] = svd(coma);

% aqui escolhemos um K que determina os melhores componentes principais
% achar um K de acordo com a variancia preservada em 0.98 (0.99 ou 1 nao apresentou um resultado mt satisfatorio)
K = 0;
for K = 1:columns(data_wc)
  diagonal = diag(eigvalues);
  calc = sum(diagonal(1:K,1))/sum(diagonal);
  if(calc >= .98)
    break;
  endif
endfor

printf("Encontramos %d componentes principais de forma que %.2f %% da variancia da base foi preservada\n", K, (calc*100));

% filtrando apenas os autovetores que importam
evreduced = eigvectors(:, 1:K);

% voltamos com a coluna de classes para a matriz final ao mesmo tempo que projetamos a base de dados no espaço do PCA reduzido pelos K melhores autovalores
data_PCA = [data_wc*evreduced, all_data(:,end)];


