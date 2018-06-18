%%% PREDIÇÃO %%%
function y_pred = neural_network(hidden_neurons, max_iter, ktrain, ktest) 
  % Implementacao de redes neurais artificiais para o problema de reconhecimento 
  % de atividades humanas

  input_layer_size = 151;
  number_of_hidden_layers = 1;
  hidden_layer_size = hidden_neurons;
  num_labels = 6;

  fprintf("Carregando pesos...\n");
  load("weights.mat");
  rna_params = [Theta1(:) ; Theta2(:)];
 
  % regularização dos pesos
  lambda = 0;

  J = nn_cost(rna_params, input_layer_size, hidden_layer_size, ...
                     num_labels, ktrain(:, 1:end-1), ktrain(:, end), lambda);

  % regularização dos custos 
  lambda = 1;

  J = nn_cost(rna_params, input_layer_size, hidden_layer_size, ...
                     num_labels, ktrain(:, 1:end-1), ktrain(:, end), lambda);
                     
                     
  % gradiente sigmoid
  g = sigmoidal_grad([1 -0.5 0 0.5 1]);

  
  initial_Theta1 = random_init(input_layer_size, hidden_layer_size);
  initial_Theta2 = random_init(hidden_layer_size, num_labels);

  initial_rna_params = [initial_Theta1(:) ; initial_Theta2(:)];

  % treinamento
  % TODO: mudar o maxiter  e o lambda pra ver como influencia no treinamento
  options = optimset('MaxIter', max_iter);
  lambda = 1;

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

%%% PREDIÇÃO %%%
function p = prediction(Theta1, Theta2, X)
  %PREDICAO Prediz o rotulo de uma amostra apresentada a rede neural
  %   p = PREDICAO(Theta1, Theta2, X) prediz o rotulo de X ao utilizar
  %   os pesos treinados na rede neural (Theta1, Theta2)

  m = size(X, 1);
  num_labels = size(Theta2, 1);

  p = zeros(size(X, 1), 1);
  
  h1 = sigmoid([ones(m, 1) X] * Theta1');
  h2 = sigmoid([ones(m, 1) h1] * Theta2');
  [dummy, p] = max(h2, [], 2);

end



%%% INICIALIZAÇÃO DOS PESOS %%%
function W = random_init(il_size, hl_size)
  % inicializa os pesos randomicamente
  % il_size : tamanho da camada de input
  % hl_size : tamanho da camada oculta
  
  % retorna uma matriz de tamanho il_size X hl_size
  % pra funcionar, tem que instalar dois pacotes
  %   pkg install io-2.4.11.tar.gz
  % e depois
  %   pkg install statistics-1.4.0.tar.gz
  
  W = stdnormal_rnd(hl_size, il_size+1);
  % W = randn(il_size, hl_size);
endfunction


%%% FUNÇÕES DE ATIVAÇÃO %%%
function R = leakly_relu(z)
  R = max(0.01*y,z);
endfunction

function R = relu(z)
  R = max(0,z);
endfunction

function g = sigmoid(z)
  g = 1.0 ./ (1.0 + exp(-z));
endfunction


%%% CALCULO DO CUSTO %%%
function [J grad] = nn_cost(nn_params, ...
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

  a1 = [ones(m,1) X];
  z2 = a1 * Theta1';
  a2 = [ones(m,1) sigmoid(z2)]; 
  z3 = a2 * Theta2';  
  a3 = sigmoid(z3);
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


%%% GRADIENTES %%%
function numgrad = numerical_grad(J, theta)
  %GRADIENTENUMERICO Calcula o gradiente usando "diferencas finitas"
  %e da um resultado estimado do gradiente.
  %   numgrad = GRADIENTENUMERICO(J, theta) calcula o gradiente numerico
  %   da funcao J acerca do theta. Executando y = J(theta) deve
  %   retorna o valor da funcao para theta.

  % Notas: O codigo a seguir implementa a checagem do gradiente numerico
  %        e retorna o gradiente numerico. O valor numgrad(i) se refere
  %        a uma aproximacao numerico da derivada parcial de J com relacao
  %        ao i-esimo argumento de entrada. Ou seja, numgrad(i) se
  %        refere a derivada parcial aproximada de J com relacao a theta(i).
  %                

  numgrad = zeros(size(theta));
  perturb = zeros(size(theta));
  e = 1e-4;

  for p = 1:numel(theta)

      % Define vetor de perturbacao
      perturb(p) = e;
      loss1 = J(theta - perturb);
      loss2 = J(theta + perturb);
    
      % Calcula o gradiente numerico
      numgrad(p) = (loss2 - loss1) / (2*e);
      perturb(p) = 0;
    
  end
end

function g = sigmoidal_grad(z)
  %GRADIENTESIGMOIDE retorna o gradiente da funcao sigmoidal para z
  g = zeros(size(z));
  g = sigmoid(z) .* (1 - sigmoid(z));
end

%%% FMINCG %%%
function [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
  % Minimize a continuous differentialble multivariate function. Starting point
  % is given by "X" (D by 1), and the function named in the string "f", must
  % return a function value and a vector of partial derivatives. The Polack-
  % Ribiere flavour of conjugate gradients is used to compute search directions,
  % and a line search using quadratic and cubic polynomial approximations and the
  % Wolfe-Powell stopping criteria is used together with the slope ratio method
  % for guessing initial step sizes. Additionally a bunch of checks are made to
  % make sure that exploration is taking place and that extrapolation will not
  % be unboundedly large. The "length" gives the length of the run: if it is
  % positive, it gives the maximum number of line searches, if negative its
  % absolute gives the maximum allowed number of function evaluations. You can
  % (optionally) give "length" a second component, which will indicate the
  % reduction in function value to be expected in the first line-search (defaults
  % to 1.0). The function returns when either its length is up, or if no further
  % progress can be made (ie, we are at a minimum, or so close that due to
  % numerical problems, we cannot get any closer). If the function terminates
  % within a few iterations, it could be an indication that the function value
  % and derivatives are not consistent (ie, there may be a bug in the
  % implementation of your "f" function). The function returns the found
  % solution "X", a vector of function values "fX" indicating the progress made
  % and "i" the number of iterations (line searches or function evaluations,
  % depending on the sign of "length") used.
  %
  % Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
  %
  % See also: checkgrad 
  %
  % Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
  %
  %
  % (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
  % 
  % Permission is granted for anyone to copy, use, or modify these
  % programs and accompanying documents for purposes of research or
  % education, provided this copyright notice is retained, and note is
  % made of any changes that have been made.
  % 
  % These programs and documents are distributed without any warranty,
  % express or implied.  As the programs were written for research
  % purposes only, they have not been tested to the degree that would be
  % advisable in any important application.  All use of these programs is
  % entirely at the user's own risk.
  %
  % [ml-class] Changes Made:
  % 1) Function name and argument specifications
  % 2) Output display
  %

  % Read options
  if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
      length = options.MaxIter;
  else
      length = 100;
  end


  RHO = 0.01;                            % a bunch of constants for line searches
  SIG = 0.5;       % RHO and SIG are the constants in the Wolfe-Powell conditions
  INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
  EXT = 3.0;                    % extrapolate maximum 3 times the current bracket
  MAX = 20;                         % max 20 function evaluations per line search
  RATIO = 100;                                      % maximum allowed slope ratio

  argstr = ['feval(f, X'];                      % compose string used to call function
  for i = 1:(nargin - 3)
    argstr = [argstr, ',P', int2str(i)];
  end
  argstr = [argstr, ')'];

  if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
  S=['Iteracao '];

  i = 0;                                            % zero the run length counter
  ls_failed = 0;                             % no previous line search has failed
  fX = [];
  [f1 df1] = eval(argstr);                      % get function value and gradient
  i = i + (length<0);                                            % count epochs?!
  s = -df1;                                        % search direction is steepest
  d1 = -s'*s;                                                 % this is the slope
  z1 = red/(1-d1);                                  % initial step is red/(|s|+1)

  while i < abs(length)                                      % while not finished
    i = i + (length>0);                                      % count iterations?!

    X0 = X; f0 = f1; df0 = df1;                   % make a copy of current values
    X = X + z1*s;                                             % begin line search
    [f2 df2] = eval(argstr);
    i = i + (length<0);                                          % count epochs?!
    d2 = df2'*s;
    f3 = f1; d3 = d1; z3 = -z1;             % initialize point 3 equal to point 1
    if length>0, M = MAX; else M = min(MAX, -length-i); end
    success = 0; limit = -1;                     % initialize quanteties
    while 1
      while ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0) 
        limit = z1;                                         % tighten the bracket
        if f2 > f1
          z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);                 % quadratic fit
        else
          A = 6*(f2-f3)/z3+3*(d2+d3);                                 % cubic fit
          B = 3*(f3-f2)-z3*(d3+2*d2);
          z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;       % numerical error possible - ok!
        end
        if isnan(z2) || isinf(z2)
          z2 = z3/2;                  % if we had a numerical problem then bisect
        end
        z2 = max(min(z2, INT*z3),(1-INT)*z3);  % don't accept too close to limits
        z1 = z1 + z2;                                           % update the step
        X = X + z2*s;
        [f2 df2] = eval(argstr);
        M = M - 1; i = i + (length<0);                           % count epochs?!
        d2 = df2'*s;
        z3 = z3-z2;                    % z3 is now relative to the location of z2
      end
      if f2 > f1+z1*RHO*d1 || d2 > -SIG*d1
        break;                                                % this is a failure
      elseif d2 > SIG*d1
        success = 1; break;                                             % success
      elseif M == 0
        break;                                                          % failure
      end
      A = 6*(f2-f3)/z3+3*(d2+d3);                      % make cubic extrapolation
      B = 3*(f3-f2)-z3*(d3+2*d2);
      z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));        % num. error possible - ok!
      if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0   % num prob or wrong sign?
        if limit < -0.5                               % if we have no upper limit
          z2 = z1 * (EXT-1);                 % the extrapolate the maximum amount
        else
          z2 = (limit-z1)/2;                                   % otherwise bisect
        end
      elseif (limit > -0.5) && (z2+z1 > limit)          % extraplation beyond max?
        z2 = (limit-z1)/2;                                               % bisect
      elseif (limit < -0.5) && (z2+z1 > z1*EXT)       % extrapolation beyond limit
        z2 = z1*(EXT-1.0);                           % set to extrapolation limit
      elseif z2 < -z3*INT
        z2 = -z3*INT;
      elseif (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT))   % too close to limit?
        z2 = (limit-z1)*(1.0-INT);
      end
      f3 = f2; d3 = d2; z3 = -z2;                  % set point 3 equal to point 2
      z1 = z1 + z2; X = X + z2*s;                      % update current estimates
      [f2 df2] = eval(argstr);
      M = M - 1; i = i + (length<0);                             % count epochs?!
      d2 = df2'*s;
    end                                                      % end of line search

    if success                                         % if line search succeeded
      f1 = f2; fX = [fX' f1]';
      fprintf('%s %4i | Custo: %4.6e\r', S, i, f1);
      s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
      tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
      d2 = df1'*s;
      if d2 > 0                                      % new slope must be negative
        s = -df1;                              % otherwise use steepest direction
        d2 = -s'*s;    
      end
      z1 = z1 * min(RATIO, d1/(d2-realmin));          % slope ratio but max RATIO
      d1 = d2;
      ls_failed = 0;                              % this line search did not fail
    else
      X = X0; f1 = f0; df1 = df0;  % restore point from before failed line search
      if ls_failed || i > abs(length)          % line search failed twice in a row
        break;                             % or we ran out of time, so we give up
      end
      tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
      s = -df1;                                                    % try steepest
      d1 = -s'*s;
      z1 = 1/(1-d1);                     
      ls_failed = 1;                                    % this line search failed
    end
    if exist('OCTAVE_VERSION')
      fflush(stdout);
    end
  end
  fprintf('\n');
end