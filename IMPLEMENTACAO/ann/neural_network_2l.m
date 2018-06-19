%%% PREDIÇÃO %%%
function y_pred = neural_network_2l(hidden_neurons, max_iter, ktrain, ktest, lambda) 
  % Implementacao de redes neurais artificiais para o problema de reconhecimento 
  % de atividades humanas

  input_layer_size = 151;
  number_of_hidden_layers = 2;
  hidden_layer_size = hidden_neurons;
  num_labels = 6;



  initial_Theta1 = random_init(input_layer_size, hidden_layer_size);
  initial_Theta2 = random_init(hidden_layer_size, hidden_layer_size);
  initial_Theta3 = random_init(hidden_layer_size, num_labels);
  
  initial_rna_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)]; 
  initial_cost = nn_cost_2l( initial_rna_params, input_layer_size, hidden_layer_size, num_labels, ktrain(:, 1:end-1),  ktrain(:, end), lambda);
  % treinamento
  % TODO: mudar o maxiter  e o lambda pra ver como influencia no treinamento
  options = optimset('MaxIter', max_iter);
  

  % Cria uma nova chamada para minimizar a funcao de custo
  cost_func = @(p) nn_cost_2l(p, ...
                                     input_layer_size, ...
                                     hidden_layer_size, ...
                                     num_labels, ktrain(:, 1:end-1), ...
                                     ktrain(:, end), lambda);

  % Agora, cost_func eh uma funcao que recebe apenas os parametros da rede neural.
  [rna_params, cost] = fmincg(cost_func, initial_rna_params, options);

  % Obtem Theta1 e Theta2 back a partir de rna_params
  Theta1 = reshape(rna_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));


  Theta2 = reshape(rna_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));
                     
  printf("Salvando resultados obtidos em disco. (Pesos inicias, custo inicial, Pesos Iterados, e custo final)");
  file_name = "./data/weighs2_";
  strcat( file_name, "_", num2str(max_iter),"iters_", num2str(lambda), "lambda.mat" );
  save(file_name, 'initial_Theta1', 'initial_Theta2', 'initial_Theta3', 'initial_cost','Theta1', 'Theta2', 'Theta3', 'cost' );
  printf("Arquivo salvo como: %s", file_name);
  
  % predicao treinamento
  fprintf('Calculando acuracia na base de treinamento...\n');
  pred = prediction_2l( Theta1, Theta2, Theta3, ktrain(:,1:end-1) );

  fprintf('Acuracia no conjunto de treinamento: %f\n', mean( double(pred == ktrain(:,end)) ) * 100);
  
  y_pred = prediction_2l(Theta1, Theta2, Theta3, ktest(:,1:end-1));
end