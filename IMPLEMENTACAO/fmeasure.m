function f = fmeasure(ypred, yreal)
  
  for i = 1:6
    tp = rows(intersect(find(ypred == i), find(yreal == i)));
    fp = rows(intersect(find(ypred == i), find(yreal != i)));
    tn = rows(intersect(find(ypred != i), find(yreal == i)));
    if tp + fp > 0
      % precisao para cada classe
      prec(i) = tp/(tp+fp);
    endif
    if tp + tn > 0
      % revocacao para cada classe
      recall(i) = tp/(tp+tn);
    endif
  end
  
  % calculo da f-medida com a media da precisao e revocacao
  f = (2 * mean(prec) * mean(recall) / (mean(prec) + mean(recall))) * 100;
end