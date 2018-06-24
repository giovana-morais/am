% Grupo 7 - Reconhecimento de Atividades Humanas
% este arquivo gera as curvas de aprendizado para cada classificador
pkg load statistics
pkg load image
if(strcmp(computer(), "x86_64-pc-linux-gnu") )
  addpath("./libsvm/libsvm_x86_64-pc-linux-gnu");
elseif(ispc())
  addpath("./libsvm/libsvm-windows");
endif

addpath("./ann");
addpath("./regression");

printf("Iniciando execucao.\n");
clear all, clc, close all;

% inicialmente ja fizemos pre-processamento dos dados para tentar diminuir a dimens�o de atributos e amostras
% retirando inconsistencias, redundancias, c�lulas nulas e fazendo a normalizacao de valores
% agora recuperamos os dados pre_processados que foram salvos no arquivo pre_processed

if(exist ("./data/pre_processed.mat.zip", "file") )
  printf("\nCarregando dados pre-processados...\n");
  fflush(stdout);
  load("./data/pre_processed.mat.zip");
else
  printf("Detectado que o pre-processamento ainda nao fora executado, aguarde, pre-processando....\n\n");
  fflush(stdout);
  preprocessing;
endif

if(exist ("./data/data_pca.mat.zip", "file") )
  load("./data/data_pca.mat.zip");
  printf("Dados do PCA Carregados!\n\n");
  fflush(stdout);
else
  # passando os dados pelo PCA para reduzir os atributos e utilizar em algoritmos que demandam mais processamento
  data_pca = pca(all_data);
  save("-zip", "./data/data_pca.mat.zip", "data_pca");
endif


% proveniente do 10-fold cross validation utilizado para a escolha de melhor fold
ksize = floor(rows(all_data)/10);

iter = 4;

ktest = all_data((((iter-1)*ksize)+1):(((iter-1)*ksize)+ksize), :);
ktrain = [all_data(1:(ksize*(iter-1)),:); all_data((((iter-1)*ksize)+ksize+1):end, :)];
ktest_pca = data_pca((((iter-1)*ksize)+1):(((iter-1)*ksize)+ksize), :);
ktrain_pca = [data_pca(1:(ksize*(iter-1)),:); data_pca(((iter-1)*ksize+ksize+1):end, :)];


k = 1;
lambda_rl = 2;
hidden_neurons_rn1 = 151;
max_iter_rn1 = 750;
lambda_rn1 = 1;
hidden_neurons_rn2 = 300;
max_iter_rn2 = 1000;
lambda_rn2 = 2;
c = 32;
g = 0.0078125;

ktam = floor((rows(all_data)-rows(ktest))/100);


for tam = 1:100
    printf("\nIncremento %d\n", tam);
    if(tam != 100)
      cur_ktrain = ktrain(1:ktam*tam, :);
      cur_ktrain_pca = ktrain_pca(1:ktam*tam, :);
    else
      cur_ktrain = ktrain(:, :);
      cur_ktrain_pca = ktrain_pca(:, :);
    endif

    % knn com melhor k encontrado com grid search ----------------------------------------------------------
    for i = 1:rows(ktest)
      ypred_knn_test(i,1) = knn(cur_ktrain(:,1:end-1), cur_ktrain(:,end), ktest(i,1:end-1), k);
    endfor

    for i = 1:rows(cur_ktrain)
      ypred_knn_train(i,1) = knn(cur_ktrain(:,1:end-1), cur_ktrain(:,end), cur_ktrain(i,1:end-1), k);
    endfor

    knncost_test(tam) = immse(ypred_knn_test, ktest(:, end));
    knncost_train(tam) = immse(ypred_knn_train, cur_ktrain(:,end));

%    % regressao logistica com melhor lambda encontrado com grid search ---------------------------------------------------
 
    ypred_rl_test = regression(cur_ktrain(:,1:end-1), cur_ktrain(:,end), ktest(:,1:end-1), lambda_rl);

    ypred_rl_train = regression(cur_ktrain(:,1:end-1), cur_ktrain(:,end), cur_ktrain(:,1:end-1), lambda_rl);
    
    rlcost_test(tam) = immse(ypred_rl_test, ktest(:, end));
    rlcost_train(tam) = immse(ypred_rl_train, cur_ktrain(:,end));

    % redes neurais com o melhor lambda, max_iter encontrados com grid search -------------------------------------
    
	  ypred_rn1_test = neural_network_1l(hidden_neurons_rn1, max_iter_rn1, cur_ktrain_pca, ktest_pca, lambda_rn1);
	  ypred_rn1_train = neural_network_1l(hidden_neurons_rn1, max_iter_rn1, cur_ktrain_pca, cur_ktrain_pca, lambda_rn1);
	
	  rn1cost_test(tam) = immse(ypred_rn1_test, ktest_pca(:, end));
    rn1cost_train(tam) = immse(ypred_rn1_train, cur_ktrain_pca(:,end));

	  ypred_rn2_test = neural_network_2l(hidden_neurons_rn2, max_iter_rn2, cur_ktrain_pca, ktest_pca, lambda_rn2);
	  ypred_rn2_train = neural_network_2l(hidden_neurons_rn2, max_iter_rn2, cur_ktrain_pca, cur_ktrain_pca, lambda_rn2);
	
	  rn2cost_test(tam) = immse(ypred_rn2_test, ktest_pca(:, end));
    rn2cost_train(tam) = immse(ypred_rn2_train, cur_ktrain_pca(:,end));

    % svm com melhor c e gamma encontrados com grid search -----------------------------------------
    
    [ypred_svm_test, ~, ~] = svm(cur_ktrain_pca(:,1:end-1), cur_ktrain_pca(:,end), ktest_pca, c, g);

    [ypred_svm_train, ~, ~] = svm(cur_ktrain_pca(:,1:end-1), cur_ktrain_pca(:,end), cur_ktrain_pca, c, g);
    
    svmcost_test(tam) = immse(ypred_svm_test, ktest_pca(:, end));
    svmcost_train(tam) = immse(ypred_svm_train, cur_ktrain_pca(:,end));
 
endfor
 
    plot_cost(knncost_test', knncost_train', 'Curva de aprendizado para o KNN');
    plot_cost(rlcost_test', rlcost_train', 'Curva de aprendizado para a Regressao Linear');
    plot_cost(rn1cost_test', rn1cost_train', 'Curva de aprendizado para Rede Neural Artificial com 1 camada oculta');
	  plot_cost(rn2cost_test', rn2cost_train', 'Curva de aprendizado para Rede Neural Artificial com 2 camadas ocultas');
    plot_cost(svmcost_test', svmcost_train', 'Curva de aprendizado para o SVM');