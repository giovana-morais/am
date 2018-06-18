function g = sigmoidal_grad(z)
  %GRADIENTESIGMOIDE retorna o gradiente da funcao sigmoidal para z
  g = zeros(size(z));
  g = sigmoid(z) .* (1 - sigmoid(z));
end