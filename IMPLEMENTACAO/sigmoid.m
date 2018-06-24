function g = sigmoid(z)
  % eh usada por rna e regressao
  g = 1.0 ./ (1.0 + exp(-z));
endfunction