
function y_pred = neural_network_1l(hidden_neurons, max_iter, ktrain, ktest,lambda) 
  % Implementacao de redes neurais artificiais para o problema de reconhecimento 
  % de atividades humanas

  input_layer_size = 151;
  number_of_hidden_layers = 1;
  hidden_layer_size = hidden_neurons;
  num_labels = 6;

  fprintf("Carregando pesos...\n");
  
  rna_params = [Theta1(:) ; Theta2(:)];
 
  % regularização dos pesos

                     
  % gradiente sigmoid
  g = sigmoidal_grad([1 -0.5 0 0.5 1]);

  
  initial_Theta1 = random_init(input_layer_size, hidden_layer_size);
  initial_Theta2 = random_init(hidden_layer_size, num_labels);

  initial_rna_params = [initial_Theta1(:) ; initial_Theta2(:)];

  % treinamento
  % TODO: mudar o maxiter  e o lambda pra ver como influencia no treinamento
  options = optimset('MaxIter', max_iter);

  % Cria uma nova chamada para minimizar a funcao de custo
  cost_func = @(p) nn_cost(p, ...
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
                     
                    
  % predicao treinamento
  fprintf('Treinando...\n');
  pred = prediction(Theta1, Theta2, ktrain(:,1:end-1));

  fprintf('Acuracia no conjunto de treinamento: %f\n', mean(double(pred == ktrain(:,end))) * 100);
  
  y_pred = prediction(Theta1, Theta2, ktest(:,1:end-1));
end





