% Grupo 7 - Reconhecimento de Atividades Humanas
pkg load statistics

printf("Iniciando execucao.\n");
clear all, clc, close all;

% inicialmente ja fizemos pre-processamento dos dados para tentar diminuir a dimens�o de atributos e amostras
% retirando inconsistencias, redundancias, c�lulas nulas e fazendo a normalizacao de valores
% agora recuperamos os dados pre_processados que foram salvos no arquivo pre_processed

printf("\nCarregando dados pre-processados...\n");
fflush(stdout);
load("pre_processed.mat");

% em seguida, devemos dividir os dados gerais para validacao cruzada, com k-fold sendo 10 (mais usualmente utilizado em AM)
% assim sendo, 9 partes irao para treinamento enquanto apenas 1 ira para teste (isso ocorre 10 vezes, para cada algoritmo)
% aqui conseguiremos avaliar as hipoeses e selecionar a melhor

ksize = floor(length(all_data)/10);

printf("\nIniciando grid do 10-fold cross-validation\n");
fflush(stdout);

% aqui inicializamos o grid (com zeros) de cada algoritmo para escolha do melhor fold posteriormente
% o tamanho das linhas representa os k-folds
% o tamanho de colunas do grid deve ser a quantidade maxima do parametro que queremos fazer o ajuste para cada algoritmo
gridknn = zeros(10, 50);
gridrl = zeros(10, 10);
gridrn = zeros(10, 4);  % considerando por enquanto só as 4 variações do max_iter
gridsvmRbf = zeros(10, 152); % 8 variacoes do C e 19 variacoes do Gamma
gridsvmLinear = zeros(10, 8); % 8 variacoes do C


for iter = 1:10
  % se o fold de teste nao for inicial nem final, quer dizer que ktest nao comeca no 1 e nem termina no end
  if iter ~= 1 && iter ~= 10
    ktest = all_data((((iter-1)*ksize)+1):(((iter-1)*ksize)+ksize), :);
    ktrain = [all_data(1:(ksize*(iter-1)),:); all_data((((iter-1)*ksize)+ksize+1):end, :)];
  elseif iter == 1
    ktest = all_data(1:ksize, :);
    ktrain = all_data((ksize+1):end, :);
  elseif iter == 10
    ktest = all_data((iter-1)*ksize:end, :);
    ktrain = all_data(1:((ksize*(iter-1))-1), :);
  endif
  
  printf("\nIteracao onde ktest eh o %d-fold e o ktrain eh o restante, o tamanho de ktest eh %d e o tamanho de ktrain eh %d\n", iter, length(ktest), length(ktrain));

  % execucao do knn
  printf('\nIniciando execucao do knn\n');
  fflush(stdout);
  % escolhemos o k com maior acuracia
  tic();
  for k = 1:50
    printf("\nPara k = %d\n", k);
    fflush(stdout);
    ac = 0;
    for i = 1:rows(ktest)
      ypred = knn(ktrain(:,1:end-1), ktrain(:,end), ktest(i,1:end-1), k);
      if(ypred == ktest(i, end))
        ac += 1;
      endif
    endfor
    acuracyknn(k) = ac/rows(ktest);
    printf("Ocorre %.2f%% de acuracia\n", acuracyknn(k)*100);
    fflush(stdout);
    
    % salvando acuracia no grid do knn
    gridknn(iter, k) = acuracyknn(k)*100;
  endfor
  toc();
  
  fprintf('\nO algoritmo KNN finalizou a execucao. \n');

  % execucao da regressao logistica
  for lambda=0:10
    printf('\nIniciando execucao da regressao logistica para lambda = %d\n', lambda);
    fflush(stdout);
    tic();
    y_pred = regression(ktrain(:,1:end-1), ktrain(:,end), ktest(:,1:end-1), lambda);
    toc();
    acc_reg = mean(double(y_pred == ktest(:,end))) * 100;
    printf('\nAcuracia do teste: %.2f\n', acc_reg);
    fflush(stdout);
    gridrl(iter, lambda+1) = acc_reg;
  end
  fprintf('\nO algoritmo de Regressao Logistica finalizou a execucao. \n');

  ####### executando o PCA para reduzir a quantidade de atributos e garantir a viabilidade de execu��o de redes neurais e do SVM (30min com PCA, 1h20 sem PCA) #####
  all_data = pca(all_data);
  
  % execucao de redes neurais
  printf('\nIniciando execucao de redes neurais artificiais\n');
  fflush(stdout);
  
  tic();
  % o número de neuronios ocultos deve variar entre
  % 1) 2/3 do tamanho da camada de entrada
  % 2) alguns números entre o tamanho da camada de entrada e o dobro dela
  hidden_neurons = [151];
  max_iter = [50, 100, 150, 300];
  
  for i = 1: length(max_iter)
    %fprintf("\nMAX ITER = %d\n", max_iter(i)); 
    y_pred = neural_network(hidden_neurons(1), max_iter(i), ktrain, ktest);
    acc_nn = mean(double(y_pred == ktest(:,end))) * 100
    gridrl(iter, i) = acc_nn;
    %fprintf('Acuracia no conjunto de teste: %f\n', acc_nn);
  end
  fprintf('\nO algoritmo de Redes Neurais Artificiais finalizou a execucao. \n');
  toc();
  
  % execucao da svm
  printf('\nIniciando execucao de SVM\n');
  fflush(stdout);
  tic();
  [ypred, gridLin, gridRbf] = svm(ktrain(:,1:end-1), ktrain(:,end), ktest);
  toc();
  fprintf('\nO algoritmo SVM finalizou a execucao. \n');
  
endfor

% estruturacao do grid 

totalgrid = [];
% aqui localizamos a coluna onde esta contido o valor maximo de acuracia do knn
[maxvalue,col] = max(max(gridknn));
printf("\nA maior acuracia do knn eh %.2f para k = %d\n", maxvalue, col);

%aqui localizamos a coluna onde esta contido o valor maximo de acuracia da regressao
[maxrl, colrl] = max(max(gridrl));
printf("\nA maior acuracia da regressao eh %.2f para lambda = %d\n", maxrl, colrl);

%aqui localizamos a coluna onde esta contido o valor maximo de acuracia de redes neurais
[maxrn, colrn] = max(max(gridrn));
printf("\nA maior acuracia de redes neurais eh %.2f para max_iter = %d\n", maxrn, colrn);

%aqui localizamos a coluna onde esta contido o valor maximo de acuracia de svm
[maxsvm, colsvm] = max(max(gridRbf));
printf("\nA maior acuracia de svm eh %.2f para coluna = %d\n", maxsvm, colsvm);

% Esta etapa matematica para descobrir quais sao os valores de C e Gamma representados pela coluna
isvm = ceil(colsvm/19);
jsvm = mod(colsvm, 19);

csvm = 2 ^ (-5 + isvm);
gammasvm = 2 ^ (-15 + jsvm);

printf("Os valores otimos para o SVM com Kernel Gaussiano sao C = %f e Gamma = %f", csvm, gammasvm);

% aqui pegamos os melhores valores de k para cada fold
totalgrid = [totalgrid, max(gridknn, [], 2)];

% aqui pegamos os melhores valores de lambda para cada fold
totalgrid = [totalgrid, max(gridrl, [], 2)];

totalgrid = [totalgrid, max(gridrn, [], 2)];

% aqui pegamos os melhores valores dos parametros de svm para cada fold
totalgrid = [totalgrid, max(griRbf, [], 2)];

% aqui geramos um csv para visualizacao no relatorio
csvwrite('gridknn.csv', gridknn);
csvwrite('gridrl.csv', gridrl);
csvwrite('gridrn.csv', gridrn);
csvwrite('gridRbf.csv', gridRbf);

####### depois de salvar a melhor coluna de cada algoritmo, devemos escolher o k-fold que otimiza a acuracia de todos #######
% aqui geramos um csv para visualizacao no relatorio
csvwrite('totalgrid.csv', totalgrid);

% para otimizar a acuracia, calculamos o k-fold que possui a maior media entre todos os algoritmos
[~, bestfold] = max(mean(totalgrid,2));

printf("\nOs algoritmos ficam melhor otimizados quando os dados de teste possuem o %d-fold e os dados de treinamento possuem o resto\n\n", bestfold);

% aqui encontramos em qual coluna do fold se encontra o melhor k
[valueknn, bestknn] = max(gridknn(bestfold,:));
printf("\nComo o melhor fold eh o %d\n\nO melhor k para o knn eh %d com acuracia de %.2f\n", bestfold, bestknn, valueknn);

[valuerl, bestrl] = max(gridrl(bestfold, :));
printf("\nO melhor lambda para a regressao eh %d com acuracia de %.2f\n", bestrl, valuerl);

[valuern, bestrn] = max(gridrn(bestfold, :));
printf("\nO melhor max_iter para redes neurais eh %d com acuracia de %.2f\n", bestrn, valuern);

printf("\nFim de execucao\n");

