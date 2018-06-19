function plotCost(costtest, costtrain, title_plot)
  
  figure, plot(1:rows(predtest), costtest, 'r');
  hold on;
  plot(1:rows(predtest), costtrain, 'b');
  hold off;
  legend('Teste', 'Treinamento');
  xlabel('Num de amostras');
  ylabel('Erro');
  title(title_plot);  
  
end