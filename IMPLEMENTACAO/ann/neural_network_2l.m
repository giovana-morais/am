%%% PREDICAO %%%
function [y_pred] = neural_network_2l(hidden_neurons, max_iter, ktrain, ktest, lambda) 
  % Implementacao de redes neurais artificiais para o problema de reconhecimento 
  % de atividades humanas

  input_layer_size = size(ktrain,2) -1;
  number_of_hidden_layers = 2;
  hidden_layer_size = hidden_neurons;
  num_labels = 6;

  % regularizacao dos pesos
  
  initial_Theta1 = random_init(input_layer_size, hidden_layer_size);
  initial_Theta2 = random_init(hidden_layer_size, hidden_layer_size);
  initial_Theta3 = random_init(hidden_layer_size, num_labels);

  % concatena pesos
  initial_rna_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)]; 
  
  % treinamento
  options = optimset('MaxIter', max_iter);
  

  % Cria uma nova chamada para minimizar a funcao de custo
  cost_func = @(p) nn_cost_2l(p, ...
                                     input_layer_size, ...
                                     hidden_layer_size, ...
                                     num_labels, ktrain(:, 1:end-1), ...
                                     ktrain(:, end), lambda);

  % Agora, cost_func eh uma funcao que recebe apenas os parametros da rede neural.
  [rna_params, cost] = fmincg(cost_func, initial_rna_params, options);

  % Obtem Theta1 e Theta2 e Theta3 back a partir de rna_params
  comeco_camada_1 = 1;
  fim_camada_1 = hidden_layer_size * (input_layer_size + 1);

  comeco_camada_2 = fim_camada_1 + 1;
  fim_camada_2 = comeco_camada_2 + hidden_layer_size * (hidden_layer_size + 1) - 1;

  comeco_camada_3 = fim_camada_2 + 1;                 
  fim_camada_3 =  comeco_camada_3 - 1 + (hidden_layer_size + 1) * num_labels;
  
  
  Theta1 = reshape(rna_params(comeco_camada_1:fim_camada_1), hidden_layer_size, input_layer_size + 1  );

  Theta2 = reshape(rna_params(comeco_camada_2:fim_camada_2), hidden_layer_size, hidden_layer_size + 1 );
  
  Theta3 = reshape(rna_params(comeco_camada_3:fim_camada_3), num_labels ,       hidden_layer_size +1  );
                  
  #printf("Salvando resultados obtidos em disco. (Pesos inicias, custo inicial, Pesos Iterados, e custo final)");
%  file_name = "./data/w2l";
%  file_name = strcat( file_name, "_", num2str(max_iter),"iters_", num2str(lambda), "lambda_", num2str(hidden_layer_size),"neurons.mat" );
%  save(file_name, 'Theta1', 'Theta2', 'Theta3', 'cost' );
%  #printf("Arquivo salvo como: %s\n\n", file_name);
  
  
  % predicao treinamento
%  fprintf('Calculando acuracia na base de treinamento...\n');
  training_prediction = prediction_2l( Theta1, Theta2, Theta3, ktrain(:,1:end-1) );

%  fprintf('Acuracia no conjunto de treinamento: %.6f%%\n', mean( double(training_prediction == ktrain(:,end)) ) * 100);
  
  y_pred = prediction_2l(Theta1, Theta2, Theta3, ktest(:,1:end-1));
end