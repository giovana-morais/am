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