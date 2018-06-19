
function [J grad] = nn_cost_2l(nn_params, ...
                             input_layer_size, ...
                             hidden_layer_size, ...
                             num_labels, ...
                             X, y, lambda)

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
  a3 = [ones(m,1) sigmoid(z3)];
  z4 = a3 * Theta3';
  a4 = sigmoid(z4);
  h_x = a4;

  % J regularizado 
  J = 1/m * sum(sum(-y_matrix.*log(h_x) - (1-y_matrix).*log(1-h_x)),2);
  J_reg = J + (lambda*(sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2)) + sum(sum(Theta3(:,2:end).^2,2)) ) /(2*m));
  J = J_reg;

  % calcula os sigmas e os deltas
  d4 = h_x .- y_matrix;
  d3 = (d4 * Theta3(:,2:end)) .* sigmoidal_grad(z3);
  d2 = (d3 * Theta2(:,2:end)) .* sigmoidal_grad(z2);
  
  
   
  delta_1 = d2' * a1;
  delta_2 = d3' * a2;
  delta_3 = d4' * a3;
  
  
  Theta1_grad = (delta_1 ./ m) + (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
  Theta2_grad = (delta_2 ./ m) + (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];
  Theta3_grad = (delta_3 ./ m) + (lambda/m)*[zeros(size(Theta3, 1), 1) Theta3(:, 2:end)];
  % junta os gradientes 
  grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];
end