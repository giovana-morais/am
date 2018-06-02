% Grupo 7 - Reconhecimento de Atividades Humanas

printf("Iniciando execucao.\n");
clear all, clc, close all;
% inicialmente ja fizemos pre-processamento dos dados para tentar diminuir a dimensao de atributos e amostras
% retirando inconsistencias, redundancias, celulas nulas e fazendo a normalizacao de valores
% agora recuperamos os dados pre_processados que foram salvos no arquivo pre_processed

printf("\nCarregando dados pre-processados...\n");
fflush(stdout);
load("pre_processed.mat");

% normalizacao
% media
m = mean(all_data(:,1:end-1));
  
% desvio padrao
s = std(all_data(:,1:end-1));
  
% calcula a norma de cada amostra
data_norm = (all_data(:,1:end-1)- m)./s;

all_data = [data_norm, all_data(:,end)];

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
gridrn = zeros(10, 1);
gridsvm = zeros(10, 1);

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
  % aqui utilizamos apenas numeros impares para nao ter que haver desempate nos k neighbours mais proximos
  for k = 1:2:50
    j = 1;
    printf("\nPara k = %d\n", k);
    fflush(stdout);
    ac = 0;
    tic();
    for i = 1:length(ktest)
      ypred = knn(ktrain(:,1:end-1), ktrain(:,end), ktest(i,1:end-1), k);
      if(ypred == ktest(i, end))
        ac += 1;
      endif
    endfor
    toc();
    acuracyknn(j) = ac/length(ktest);
    printf("Ocorre %.2f%% de acuracia\n", acuracyknn(j)*100);
    fflush(stdout);
    
    % salvando acuracia no grid do knn
    gridknn(iter, k) = acuracyknn(j)*100;
    
    j += 1;
  endfor

  fprintf('\nO algoritmo KNN finalizou a execucao. Pressione enter para continuar.\n');
  %pause;
  
  % execucao da regressao logistica
  for lambda=0:10
    printf('\nIniciando execucao da regressao logistica para lambda = %d\n', lambda);
    fflush(stdout);
    
    y_pred = regression(ktrain(:,1:end-1), ktrain(:,end), ktest(:,1:end-1), lambda);
    
    acc_reg = mean(double(y_pred == ktest(:,end))) * 100;
    % printf('\nAcurácia do teste: %f\n', acc_reg);
    % fflush(stdout);
    gridrl(iter, lambda+1) = acc_reg;
  end
  fprintf('\nO algoritmo de Regressao Logistica finalizou a execucao. Pressione enter para continuar.\n');
  %pause;

  % execucao da redes neurais
  %printf('\nIniciando execucao de redes neurais artificiais\n');
  %fflush(stdout);

  %fprintf('\nO algoritmo de Redes Neurais Artificiais finalizou a execucao. Pressione enter para continuar.\n');
  %pause;

  % execucao da svm
  printf('\nIniciando execucao de SVM\n');
  fflush(stdout);
  
   ypred = svm(ktrain(:,1:end-1), ktrain(:,end), ktest, iter);

  fprintf('\nO algoritmo SVM finalizou a execucao. Pressione enter para continuar.\n');
  %pause;
  
endfor
totalgrid = [];
% aqui localizamos a coluna onde esta contido o valor maximo de acuracia do knn
[maxvalue,col] = max(max(gridknn));
printf("\nA maior acuracia do knn eh %.2f para k = %d\n", maxvalue, col);

[maxrl, colrl] = max(max(gridrl));
printf("\nA maior acuracia da regressão eh %.2f para lambda = %d\n", maxrl, colrl);

% pegamos a a coluna do gridknn que contem o maior valor de acuracia do knn e atribuimos todos os valores dessa coluna 
% (ou seja, para todos os folds) ao totalgrid
totalgrid = [totalgrid, gridknn(:, col)];
totalgrid = [totalgrid, gridrl(:, colrl)];

% aqui geramos um csv para visualizacao no relatorio
csvwrite('gridknn.csv', gridknn);
csvwrite('gridrl.csv', gridrl);


####### depois de salvar a melhor coluna de cada algoritmo, devemos escolher o k-fold que otimiza a acuracia de todos #######
% aqui geramos um csv para visualizacao no relatorio
csvwrite('totalgrid.csv', totalgrid);

% para otimizar a acuracia, calculamos o k-fold que possui a maior media entre todos os algoritmos
[~, bestk] = max(mean(totalgrid,2));

printf("\nOs algoritmos ficam melhor otimizados quando os dados de teste possuem o %d-fold e os dados de treinamento possuem o resto\n\n", bestk);
printf("\nFim de execucao\n");

