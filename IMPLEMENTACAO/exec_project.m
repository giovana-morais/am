% Grupo 7 - Reconhecimento de Atividades Humanas
pkg load statistics

if(strcmp(computer(), "x86_64-pc-linux-gnu") )
  addpath(strcat(pwd(),"/libsvm/libsvm_x86_64-pc-linux-gnu"));
elseif(ispc())
  addpath(strcat(pwd(),"/libsvm/libsvm-windows"));
endif

addpath(strcat(pwd(),"/ann"));
addpath(strcat(pwd(),"/regression"));

printf("Iniciando execucao.\n");
clear all, clc, close all;

% inicialmente ja fizemos pre-processamento dos dados para tentar diminuir a dimensao de atributos e amostras
% retirando inconsistencias, redundancias, celulas nulas e fazendo a normalizacao de valores
% agora recuperamos os dados pre_processados que foram salvos no arquivo pre_processed

if(exist ("./data/pre_processed.mat.zip", "file") )
  printf("\nCarregando dados pre-processados...\n");
  fflush(stdout);
  load("./data/pre_processed.mat.zip");
else
  printf("Detectado que o pre-processamento ainda nao fora executado, aguarde, pre-processando....\n\n");
  preprocessing;
endif

if(exist ("./data/data_pca.mat.zip", "file") )
  load("./data/data_pca.mat.zip");
  printf("Dados com PCA Carregados!\n\n");
else
  # passando os dados pelo PCA para reduzir os atributos e utilizar em algoritmos que demandam mais processamento
  data_pca = pca(all_data);
  save("-zip", "./data/data_pca.mat.zip", "data_pca");
endif


% em seguida, devemos dividir os dados gerais para validacao cruzada, com k-fold sendo 10 (mais usualmente utilizado em AM)
% assim sendo, 9 partes irao para treinamento enquanto apenas 1 ira para teste (isso ocorre 10 vezes, para cada algoritmo)
% aqui conseguiremos avaliar as hipoeses e selecionar a melhor

ksize = floor(length(all_data)/10);
printf("\n\nPara as seguintes questoes responda 1 para sim, 0 para nao:\n");
rodar_knn = input("Gostaria de executar o KNN ?\nResposta: ");
rodar_ann1 = input("Gostaria de executar o ANN de uma camada?\nResposta: ");
rodar_ann2 = input("Gostaria de executar o ANN de duas camadas?\nResposta: ");
rodar_svm = input("Gostaria de executar o SVM ?\nResposta: ");
rodar_rl =  input("Gostaria de executar o  RL ?\nResposta: ");
printf("\nIniciando grid do 10-fold cross-validation\n");
fflush(stdout);

% aqui inicializamos o grid (com zeros) de cada algoritmo para escolha do melhor fold posteriormente
% o tamanho das linhas representa os k-folds
% o tamanho de colunas do grid deve ser a quantidade maxima do parametro que queremos fazer o ajuste para cada algoritmo
gridknn = zeros(10, 50);
gridrl = zeros(10, 10);
gridrn1 = zeros(10, 36);  % considerando 2 max_iter por 3 hidden_neurons
gridrn2 = zeros(10, 36);  % considerando 2 max_iter por 3 hidden_neurons
gridsvm = []; % 8 variacoes do C e 19 variacoes do Gamma


for iter = 1:10
  % se o fold de teste nao for inicial nem final, quer dizer que ktest nao comeca no 1 e nem termina no end
  if iter ~= 1 && iter ~= 10
    ktest = all_data((((iter-1)*ksize)+1):(((iter-1)*ksize)+ksize), :);
    ktrain = [all_data(1:(ksize*(iter-1)),:); all_data((((iter-1)*ksize)+ksize+1):end, :)];
    
    ktest_pca = data_pca((((iter-1)*ksize)+1):(((iter-1)*ksize)+ksize), :);
    ktrain_pca = [data_pca(1:(ksize*(iter-1)),:); data_pca((((iter-1)*ksize)+ksize+1):end, :)];
  elseif iter == 1
    ktest = all_data(1:ksize, :);
    ktrain = all_data((ksize+1):end, :);
    
    ktest_pca = data_pca(1:ksize, :);
    ktrain_pca = data_pca((ksize+1):end, :);
  elseif iter == 10
    ktest = all_data((iter-1)*ksize:end, :);
    ktrain = all_data(1:((ksize*(iter-1))-1), :);
    
    ktest_pca = data_pca((iter-1)*ksize:end, :);
    ktrain_pca = data_pca(1:((ksize*(iter-1))-1), :);
  endif
  
  printf("\nIteracao onde ktest eh o %d-fold e o ktrain eh o restante, o tamanho de ktest eh %d e o tamanho de ktrain eh %d\n", iter, length(ktest), length(ktrain));

  % execucao do knn --------------------------------------------------------------------------------------------
  if(rodar_knn)
    printf('\nIniciando execucao do knn\n');
    fflush(stdout);
    % escolhemos o k com maior acuracia
    for k = 1:50
      printf("\nPara k = %d\n", k);
      fflush(stdout);
      ac = 0;
      for i = 1:rows(ktest)
        ypred(i, 1) = knn(ktrain(:,1:end-1), ktrain(:,end), ktest(i,1:end-1), k);
      endfor
      accknn = mean(double(ypred == ktest(:,end))) * 100;
      printf("Ocorre %.2f%% de acuracia\n", accknn);
      fflush(stdout);
      
      fknn = fmeasure(ypred, ktest(:,end));
      printf("\nOcorre %.2f%% de F-medida\n", fknn);
      fflush(stdout);
      
      % salvando f-medida no grid do knn
      gridknn(iter, k) = fknn;
    endfor
    
    fprintf('\nO algoritmo KNN finalizou a execucao. \n');
   endif

  % execucao da regressao logistica --------------------------------------------------------------------------------------
  if(rodar_rl)
    for lambda=0:10
      printf('\nIniciando execucao da regressao logistica para lambda = %d\n', lambda);
      fflush(stdout);
      tic();
      y_pred = regression(ktrain(:,1:end-1), ktrain(:,end), ktest(:,1:end-1), lambda);
      toc();
      acc_reg = mean(double(y_pred == ktest(:,end))) * 100;
      printf("Ocorre %.2f%% de acuracia\n", acc_reg);
      fflush(stdout);
      
      freg = fmeasure(y_pred, ktest(:,end));
      printf("\nOcorre %.2f%% de F-medida\n", freg);
      fflush(stdout);
      
      gridrl(iter, lambda+1) = freg;
    end
    fprintf('\nO algoritmo de Regressao Logistica finalizou a execucao. \n');
  endif

  % execucao de redes neurais ---------------------------------------------------------------------------------------------------------
  if(rodar_ann1 || rodar_ann2) 
    hidden_neurons = [75, 151, 300];
    lambda = [0,0.5,1,2];
    max_iter_rn1 = [200,500,750];
    totaliter = length(max_iter_rn1) * length(hidden_neurons) * length(lambda);
  endif
  % execucao de redes neurais de uma camada --------------------------------------------------------------------------------------------
  if(rodar_ann1)
    printf('\nIniciando execucao de redes neurais artificiais com 1 camada\n');
    fflush(stdout);

    tic();
    count = 1;
    for i = 1: length(hidden_neurons) 
      for j = 1: length(lambda)
        for k = 1: length(max_iter_rn1)
          printf("\nIteracao %d de %d.\n", count ,totaliter );
          printf("\n--------------------\n");
          printf("NEURONIOS \t%d\n", hidden_neurons(i));
          printf("LAMBDA:   \t%d\n", lambda(j));
          printf("MAX_ITER: \t%d\n", max_iter_rn1(k));
          printf("--------------------\n");
          y_pred = neural_network_1l(hidden_neurons(i), max_iter_rn1(k), ktrain_pca, ktest_pca, lambda(j));
          
          acc_nn = mean(double(y_pred == ktest_pca(:,end))) * 100;
          printf("Ocorre %.2f%% de acuracia na base de teste\n", acc_nn);
          fflush(stdout);
          
          fnn = fmeasure(y_pred, ktest_pca(:,end));
          printf("Ocorre %.2f%% de F-medida na base de teste\n", fnn);
          fflush(stdout);
          
          gridrn1(iter, count++) = fnn;
        endfor 
      endfor
    endfor
    fprintf('\nO algoritmo de Redes Neurais Artificiais finalizou a execucao. \n');
    toc();
    
    csvwrite('gridrn1.csv', gridrn1);
    best_rn1 = max(gridrn1, [], 2);
    save("./data/best_rn1.mat", "best_rn1");
    csvwrite("./data/best_rn1.csv", best_rn1);
  endif
  
  % execucao de redes neurais de duas camada --------------------------------------------------------------------------------------------
  if(rodar_ann2)
    max_iter_rn2 = [500,750,1000];
    printf('\nIniciando execucao de redes neurais artificiais com 2 camadas\n');
    fflush(stdout);
    
    tic();
    
    
    count = 1;
    for i = 1: length(hidden_neurons) 
      for j = 1: length(lambda)
        for k = 1: length(max_iter_rn2)
          printf("\nIteracao %d de %d.\n", count ,totaliter );
          printf("--------------------\n");
          printf("NEURONIOS \t%d\n", hidden_neurons(i));
          printf("LAMBDA:   \t%d\n", lambda(j));
          printf("MAX_ITER: \t%d\n", max_iter_rn2(k));
          printf("--------------------\n");
        
          y_pred = neural_network_2l(hidden_neurons(i), max_iter_rn2(k), ktrain_pca, ktest_pca, lambda(j));
          
          acc_nn = mean(double(y_pred == ktest_pca(:,end))) * 100;
          printf('Acuracia no conjunto de teste: %f', acc_nn);
          fflush(stdout);
          
          fnn = fmeasure(y_pred, ktest_pca(:,end));
          printf("\nOcorre %.2f%% de F-medida\n", fnn);
          fflush(stdout);
          
          gridrn2(iter, count++) = fnn;
          
        endfor
      endfor
    endfor
    fprintf('\nO algoritmo de Redes Neurais Artificiais finalizou a execucao. \n');
    toc();
    
    csvwrite('gridrn2.csv', gridrn2);
    best_rn2 = max(gridrn2, [], 2);
    save("./data/best_rn2.mat", "best_rn2");
    csvwrite("./data/best_rn2.csv", best_rn2);
  endif
  
  
  % execucao da svm ----------------------------------------------------------------------------------------------------
  if(rodar_svm)
    printf('\nIniciando execucao de SVM\n');
    fflush(stdout);
    tic();
    [~, ~, gridRbf] = svm(ktrain_pca(:,1:end-1), ktrain_pca(:,end), ktest_pca);
    gridsvm = [gridsvm; gridRbf];
    toc();
    fprintf('\nO algoritmo SVM finalizou a execucao. \n');
    
    csvwrite('gridsvm.csv', gridsvm);
    best_svm = max(gridsvm, [], 2);
    save("./data/best_svm.mat", "best_svm");
    csvwrite("./data/best_svm.csv", best_svm);
   endif
endfor

% estruturacao do grid -------------------------------------------------------------------------------------------------
totalgrid = [];

% aqui geramos um csv para visualizacao no relatorio
if(rodar_knn)
%% aqui localizamos a coluna onde esta contido o valor maximo de acuracia do knn
  [maxvalue,col] = max(max(gridknn));
  printf("\nA maior F-medida do knn eh %.2f para k = %d\n", maxvalue, col);

  csvwrite('./data/gridknn.csv', gridknn);
  % aqui pegamos os melhores valores de k para cada fold
  best_knn = max(gridknn, [], 2);
  totalgrid = [totalgrid best_knn];
  save("./data/best_knn.mat", "best_knn");
  csvwrite("./data/bestknn.csv", best_knn); 
endif

if(rodar_rl)
%%aqui localizamos a coluna onde esta contido o valor maximo de acuracia da regressao
  [maxrl, colrl] = max(max(gridrl));
  printf("\nA maior F-medida da regressao eh %.2f para lambda = %d\n", maxrl, colrl);

  csvwrite('gridrl.csv', gridrl);
  % aqui pegamos os melhores valores de lambda para cada fold
  best_rl = max(gridrl, [], 2);
  totalgrid = [totalgrid, best_rl];
  save("./data/best_rl.mat", "best_rl");
  csvwrite("./data/best_rl.csv", best_rl);
endif  

if(rodar_ann1)
%%aqui localizamos a coluna onde esta contido o valor maximo de acuracia de redes neurais
  [maxrn1, colrn1] = max(max(gridrn1));
  printf("\nA maior F-medida de redes neurais eh %.2f", maxrn1, colrn1);

  csvwrite('gridrn1.csv', gridrn1);
  best_rn1 = max(gridrn1, [], 2);
  totalgrid = [totalgrid best_rn1];
  save("./data/best_rn1.mat", "best_rn1");
  csvwrite("./data/best_rn1.csv", best_rn1);
endif

if(rodar_ann2) 
  [maxrn2, colrn2] = max(max(gridrn2));
  printf("\nA maior F-medida de redes neurais eh %.2f", maxrn2, colrn2);
   
  csvwrite('gridrn2.csv', gridrn2);
  best_rn2 = max(gridrn2, [], 2);
  totalgrid = [totalgrid best_rn2];
  save("./data/best_rn2.mat", "best_rn2");
  csvwrite("./data/best_rn2.csv", best_rn2);
endif  

if(rodar_svm)
  %aqui localizamos a coluna onde esta contido o valor maximo de acuracia de svm
  [maxsvm, colsvm] = max(max(gridsvm));
  printf("\nA maior F-medida de svm eh %.2f para coluna = %d\n", maxsvm, colsvm);

  % Esta etapa matematica para descobrir quais sao os valores de C e Gamma representados pela coluna
  isvm = ceil(colsvm/19);
  jsvm = mod(colsvm, 19);

  csvm = 2 ^ (-5 + isvm);
  gammasvm = 2 ^ (-15 + jsvm);

  printf("\nOs valores otimos para o SVM com Kernel Gaussiano sao C = %f e Gamma = %f\n", csvm, gammasvm);
  
  csvwrite('gridsvm.csv', gridsvm);
  best_svm = max(gridsvm, [], 2);
  totalgrid = [totalgrid best_svm];
  save("./data/best_svm.mat", "best_svm");
  csvwrite("./data/best_svm.csv", best_svm);
endif  

####### depois de salvar a melhor coluna de cada algoritmo, devemos escolher o k-fold que otimiza a acuracia de todos #######
%aqui geramos um csv para visualizacao no relatorio
csvwrite('totalgrid.csv', totalgrid);

% para otimizar a acuracia, calculamos o k-fold que possui a maior media entre todos os algoritmos
[~, bestfold] = max(mean(totalgrid,2));

if(rodar_knn)
%% aqui encontramos em qual coluna do fold se encontra o melhor k
  [valueknn, bestknn] = max(gridknn(bestfold,:));
  printf("\nComo o melhor fold eh o %d\n\nO melhor k para o knn eh %d com F-medida de %.2f\n", bestfold, bestknn, valueknn);

endif

if(rodar_rl)
  [valuerl, bestrl] = max(gridrl(bestfold, :));
  printf("\nO melhor lambda para a regressao eh %d com F-medida de %.2f\n", bestrl, valuerl);

endif

if(rodar_ann1)
  [valuern1, bestrn1] = max(gridrn1(bestfold, :));
  [a, b, c] = r1_to_r3(bestrn1, rows(max_iter_rn1), rows(lambda), rows(hidden_neurons));
  best_maxiter_rn1 = max_iter_rn1(a);
  best_lambda_rn1 = lambda(b);
  best_neurons_rn1 = hidden_neurons(c);
  printf("\nPara redes neurais com 1 camada, o lambda eh %.2f, o max_iter eh %d e o num de neuronios eh %d com F-medida de %.2f\n", best_lambda_rn1, best_maxiter_rn1, best_neurons_rn1, valuern1);

endif

if(rodar_ann2)
  [valuern2, bestrn2] = max(gridrn2(bestfold, :));
  [a, b, c] = r1_to_r3(bestrn2, rows(max_iter_rn2), rows(lambda), rows(hidden_neurons));
  best_maxiter_rn2 = max_iter_rn2(a);
  best_lambda_rn2 = lambda(b);
  best_neurons_rn2 = hidden_neurons(c);
  printf("\nPara redes neurais com 2 camadas, o lambda eh %.2f, o max_iter eh %d e o num de neuronios eh %d com F-medida de %.2f\n", best_lambda_rn1, best_maxiter_rn1, best_neurons_rn1, valuern1);

endif

if(rodar_svm)
  [valuesvm, bestsvm] = max(gridsvm(bestfold, :));
  isvm = ceil(bestsvm/19); 
  jsvm = mod(bestsvm, 19);
  csvm = 2^(-6+isvm);
  gammasvm = 2^(-15+jsvm);
  printf("\n O melhor C e Gamma para SVM eh %d e %d com F-medida de %.2f\n", csvm, gammasvm, valuern);
endif

printf("\nFim de execucao\n");

