function [f, prec, recall] = fmeasure(ypred, yreal)
  
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
  
  prec = mean(prec);
  recall = mean(recall);
  
  % calculo da f-medida com a media da precisao e revocacao
  f = (2 * prec * recall / (prec + recall)) * 100;
end