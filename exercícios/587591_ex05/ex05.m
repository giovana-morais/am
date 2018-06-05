%% Universidade Federal de Sao Carlos - UFSCar, Sorocaba
%
%  Disciplina: Aprendizado de Maquina
%  Prof. Tiago A. Almeida
%
%  Exercicio 5 - Redes Neurais Artificiais
%
%  Instrucoes
%  ------------
% 
%  Este arquivo contem o codigo que ajudara voce a preparar e executar
%  sua rede neural. Para isso, voce precisar completar as seguites funcoes:
%
%     gradienteSigmoide.m
%     rnaCusto.m
%
%  Nao eh necessario alterar nada neste arquivo.
%

%% Inicializacao
clear ; close all; clc

%% Parametros a serem utilizados neste exercicio
input_layer_size  = 400;  % 20x20 dimensao das imagens de entrada
hidden_layer_size = 25;   % 25 neuronios na camada oculta
num_labels = 10;          % 10 rotulos, de 1 a 10   
                          % (observe que a classe "0" recebe o rotulo 10)

%% =========== Parte 1: Carregando e visualizando os dados =============
%  O exercicio comeca com a visualizacao do conjunto de dados.
%  O conjunto de dados que voce trabalhara sera de digitos manuscritos.
%

% Carregando dados de treinamento
fprintf('Carregando e visualizando os dados...\n')

load('ex05_data1.mat');
m = size(X, 1);

% Seleciona 100 amostras aleatorias para exibir
sel = randperm(size(X, 1));
sel = sel(1:100);

visualizaDados(X(sel, :));

fprintf('Programa pausado. Pressione enter para continuar.\n');
pause;


%% ================ Parte 2: Carregando os parametros ================
% Nesta parte do exercicio, sao carregados pesos pre-treinados
% para a rede neural.

fprintf('\nCarregando parametros salvos da rede neural...\n')

% Carrega os pesos nas variaveis Theta1 e Theta2
load('ex05_pesos.mat');
rna_params = [Theta1(:) ; Theta2(:)];

%% ================ Parte 3: Calcula o custo (Feedforward) ================
%  Para a rede neural, primeiro se inicia implementando a parte de feedforward
%  que retorna o custo somente. Voce devera completar o codigo na funcao
%  rnaCusto.m para que isso aconteca. Apos terminar a etapa de feedforward,
%  voce podera verificar se sua implementacao esta correta, checando se
%  sua implementacao retorna o mesmo custo descrito no codigo.
%
%  Sugere-se implementar essa etapa sem regularizacao primeiro,
%  facilitando a etapa de analise. Posteriormente, na parte 4, voce implementara
%  o custo regularizado.
%
fprintf('\nEtapa de feedforward na rede neural...\n')

% Parametro de regularizacao dos pesos (aqui sera zero).
lambda = 0;

J = rnaCusto(rna_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Custo com os parametros (carregados do arquivo ex05_pesos): %f '...
         '\n(este valor deve ser proximo de 0.287629)\n'], J);

fprintf('\nPrograma pausado. Pressione enter para continuar.\n');
pause;

%% =============== Parte 4: Regularizacao ===============
%  Quando sua funcao de custo estiver correta, voce devera implementar
%  a regularizacao no custo.
%

fprintf('\nChecando funcao de custo (c/ regularizacao) ... \n')

% Parametro de regularizacao dos pesos (aqui sera um).
lambda = 1;

J = rnaCusto(rna_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Custo com os parametros (carregados do arquivo ex05_pesos): %f '...
         '\n(este valor deve ser proximo de 0.383770)\n'], J);

fprintf('Programa pausado. Pressione enter para continuar.\n');
pause;


%% ================ Parte 5: Gradiente  ================
%  Antes de comecar a implementar a rede neural, sera necessario
%  implementar o gradiente para funcao de sigmoide. Voce devera completar
%  o codigo no arquivo gradienteSigmoide.m.
%

fprintf('\nAvaliando gradiente da sigmoide...\n')

g = gradienteSigmoide([1 -0.5 0 0.5 1]);
fprintf('Gradiente sigmoide avaliado em [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Programa pausado. Pressione enter para continuar.\n');
pause;


%% ================ Parte 6: Inicializacao dos parametros ================
%  Nesta parte comeca a implementacao das duas camadas da rede neural para
%  classificacao dos digitos manuscritos. Os pesos sao inicializados aleatoriamente.

fprintf('\nInicializando parametros para a rede neural...\n')

initial_Theta1 = inicializaPesosAleatorios(input_layer_size, hidden_layer_size);
initial_Theta2 = inicializaPesosAleatorios(hidden_layer_size, num_labels);

initial_rna_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Parte 7: Backpropagation =================
%  Neste ponto voce deve implementar o algoritmo de backpropagation.
%  Esse algoritmo devera ser adicionado na funcao rnaCusto.m para
%  retornar as derivadas parciais dos parametros.
%
fprintf('\nChecando Backpropagation... \n');

%  Verifica o gradiente usando a funcao verificaGradiente
verificaGradiente;

fprintf('\nPrograma pausado. Pressione enter para continuar.\n');
pause;


%% =============== Parte 8: Outra parte da regularizacao ===============
%  Terminado de implementar o algoritmo de backpropagation,
%  voce deve implementar a regularizacao no custo e gradiente.
%

fprintf('\nVerificando backpropagation (c/ regularizacao) ... \n')

%  Verifica gradiente usando a funcao verificaGradiente
lambda = 3;
verificaGradiente(lambda);

% Exibe os valores da funcao de custo calculado e esperado
debug_J  = rnaCusto(rna_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCusto esperado para os parametros carregados (c/ lambda = 3): %f ' ...
         '\n(este valor deve ser proximo de 0.576051)\n\n'], debug_J);

fprintf('Programa pausado. Pressione enter para continuar.\n');
pause;


%% =================== Parte 8: Treinando a rede neural ===================
%  Neste ponto, todo o codigo necessario para treinar a rede esta pronto.
%  Aqui sera utilizada a funcao fmincg, muito similar a fminunc.
%  Trata-se de um otimizador avancado capaz de treinar as funcoes de custo
%  de forma eficiente utilizando os gradientes calculados.
%
fprintf('\nTreinando a rede neural... \n')

%  Apos ter completado toda a tarefa, mude o parametro MaxIter para
%  um valor maior e verifique como isso afeta o treinamento.
options = optimset('MaxIter', 50);

%  Voce tambem pode testar valores diferentes para lambda.
lambda = 1;

% Cria uma nova chamada para minimizar a funcao de custo
funcaoCusto = @(p) rnaCusto(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Agora, funcaoCusto eh uma funcao que recebe apenas os parametros da rede neural.
[rna_params, cost] = fmincg(funcaoCusto, initial_rna_params, options);

% Obtem Theta1 e Theta2 back a partir de rna_params
Theta1 = reshape(rna_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(rna_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Programa pausado. Pressione enter para continuar.\n');
pause;


%% ================= Parte 9: Visualizando os pesos =================
%  Voce pode visualizar os pesos aprendidos pela rede neural
%  e entender que tipo de informacao a rede está capturando
%  a partir dos dados apresentados.

fprintf('\nVisualizando a rede neural... \n')

visualizaDados(Theta1(:, 2:end));

fprintf('\nPrograma pausado. Pressione enter para continuar.\n');
pause;

%% ========================== Parte 10: Predicao ==========================
%  Apos treinar a rede neural, ela sera utilizada para predizer
%  os rotulos das amostras. Voce devera implementar a funcao de "predicao"
%  para que a rede neural seja capaz de prever os rotulos no conjunto de dados
%  e calcular a acuracia do metodo.

pred = predicao(Theta1, Theta2, X);

fprintf('\nAcuracia no conjunto de treinamento: %f\n', mean(double(pred == y)) * 100);


