function [J grad] = nn_cost_1l(nn_params, ...
                             input_layer_size, ...
                             hidden_layer_size, ...
                             num_labels, ...
                             X, y, lambda)
  % Funcao de custo para rede neural artificial de 1 camada oculta
  
  % Extrai os parametros de nn_params e alimenta as variaveis Theta1 e Theta2.
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));

  m = size(X, 1);        
  
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  
  % cria uma 'matriz de permutacao' que é na vdd uma matriz que é 1 só na classe
  % da amostra avaliada
  % ref: https://octave.org/doc/v4.0.1/Special-Utility-Matrices.html 
  y_matrix = eye(num_labels)(y,:);

  %feedforward
  a1 = [ones(m,1) X];
  z2 = a1 * Theta1';
  a2 = [ones(m,1) sigmoid(z2)]; 
  z3 = a2 * Theta2';  
  a3 = sigmoid(z3);
  h_theta = a3;
  
  % J regularizado 
  J = 1/m * sum(sum(-y_matrix.*log(h_theta+eps) - (1-y_matrix).*log(1-h_theta+eps)),2);
  J_reg = J + (lambda*(sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2))) /(2*m));
  J = J_reg;

  %backpropagation
  % calcula os sigmas e os deltas
  sigma_3 = a3 .- y_matrix;
  sigma_2 = (sigma_3 * Theta2(:,2:end)) .* sigmoidal_grad(z2);
   
  delta_1 = sigma_2' * a1;
  delta_2 = sigma_3' * a2;

  Theta1_grad = (delta_1 ./ m) + (lambda/m)*Theta1;
  Theta2_grad = (delta_2 ./ m) + (lambda/m)*Theta2;

  % junta os gradientes 
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
end



