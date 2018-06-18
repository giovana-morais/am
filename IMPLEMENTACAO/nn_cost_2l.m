function [J grad] = nn_cost_2l(nn_params, ...
                             input_layer_size, ...
                             hidden_layer_size, ...
                             num_labels, ...
                             X, y, lambda)
  %RNACUSTO Implementa a funcao de custo para a rede neural com duas camadas
  %voltada para tarefa de classificacao
  %   [J grad] = RNACUSTO(nn_params, hidden_layer_size, num_labels, ...
  %   X, y, lambda) calcula o custo e gradiente da rede neural. The
  %   Os parametros da rede neural sao colocados no vetor nn_params
  %   e precisam ser transformados de volta nas matrizes de peso.
  %
  %   input_layer_size - tamanho da camada de entrada
  %   hidden_layer_size - tamanho da camada oculta
  %   num_labels - numero de classes possiveis
  %   lambda - parametro de regularizacao
  %
  %   O vetor grad de retorno contem todas as derivadas parciais
  %   da rede neural.
  %

  % Extrai os parametros de nn_params e alimenta as variaveis Theta1 e Theta2.
  
  comeco_camada_1 = 1;
  fim_camada_1 = hidden_layer_size * (input_layer_size + 1);

  comeco_camada_2 = fim_camada_1 + 1;
  fim_camada_2 = comeco_camada_2 + hidden_layer_size * (hidden_layer_size + 1) - 1;

  comeco_camada_3 = fim_camada_2 + 1;                 
  fim_camada_3 =  comeco_camada_3 - 1 + (hidden_layer_size + 1) * num_labels;
  
  Theta1 = reshape(nn_params(comeco_camada_1:fim_camada_1), hidden_layer_size, input_layer_size + 1  );

  Theta2 = reshape(nn_params(comeco_camada_2:fim_camada_2), hidden_layer_size, hidden_layer_size + 1 );
  
  Theta3 = reshape(nn_params(comeco_camada_3:fim_camada_3), num_labels ,       hidden_layer_size +1  );
  
  
  m = size(X, 1);        
  
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  Theta3_grad = zeros(size(Theta3));
  % cria uma 'matriz de permutacao' que é na vdd uma matriz que é 1 só na classe
  % da amostra avaliada
  % ref: https://octave.org/doc/v4.0.1/Special-Utility-Matrices.html 
  y_matrix = eye(num_labels)(y,:);

  a1 = [ones(m,1) X];
  z2 = a1 * Theta1';
  a2 = [ones(m,1) sigmoid(z2)]; 
  z3 = a2 * Theta2';  
  a3 = sigmoid(z3);
  z4 = a3 * Theta3';
  a4 = sigmoid(z4);
  h_theta = a3;

  % J regularizado 
  J = 1/m * sum(sum(-y_matrix.*log(h_theta) - (1-y_matrix).*log(1-h_theta)),2);
  J_reg = J + lambda*(sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2))) /(2*m);
  J = J_reg;

  % calcula os sigmas e os deltas
  sigma_3 = a3 .- y_matrix;
  sigma_2 = (sigma_3 * Theta2) .* sigmoidal_grad([ones(size(z2, 1), 1) z2]);
  size(sigma_2);
  sigma_2 = sigma_2(:, 2:end);
   
  delta_1 = sigma_2' * a1;
  delta_2 = sigma_3' * a2;

  Theta1_grad = (delta_1 ./ m) + (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
  Theta2_grad = (delta_2 ./ m) + (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

  % junta os gradientes 
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
end