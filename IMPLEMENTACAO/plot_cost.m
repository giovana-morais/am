function plotCost(costtest, costtrain, title_plot)
  
  figure, plot(1:rows(costtest), costtest, 'r');
  hold on;
  plot(1:rows(costtest), costtrain, 'b');
  hold off;
  legend('Teste', 'Treinamento');
  xlabel('Incremento do num de amostras');
  ylabel('Erro quadratico medio');
  title(title_plot);  
  
end