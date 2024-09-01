rm(list = ls())
Calculation <- function(l) {
  source_python('StochasticOTDiscrete.py')
  n = 200
  p = 1000
  data = GendataLDA(n, p, 3, error = "gaussian", style = "unbalanced")
  
  omega_W = matrix(NA, 1, p)
  Y = data$Y
  dataX = data$X
  facindex = c()
  for (i in c(1:p)) {
    print(i)
    STD = cbind(Y, dataX[, i])
    Y = as.factor(Y)
    tryCatch({
      X = dataX[, i]
      bw = npudensbw(formula =  ~ Y + X)
      bwy = npudensbw(~ Y)
      bwx = npudensbw(~ X)
      
      fyx = npudens(bws = bw)
      fy = npudens(bws = bwy)
      fx = npudens(bws = bwx)
      
      jointdens = fyx$dens
      margdens = fy$dens * fx$dens
      
      jointdens = matrix(jointdens / sum(jointdens), n, 1)
      margdens = matrix(margdens / sum(margdens), n, 1)
      
      
      discreteOTComputer = PyTorchStochasticDiscreteOT(
        STD,
        jointdens,
        STD,
        margdens,
        reg_type = 'l2',
        reg_val = 0.1,
        device_type = 'cpu'
      )
      history = discreteOTComputer$learn_OT_dual_variables(epochs = 200,
                                                           batch_size = 50,
                                                           lr = 0.0005)
      omega_W[1, i] = discreteOTComputer$compute_OT_MonteCarlo(epochs =
                                                                 20, batch_size = 50)
    }
  },
  error = function(e) {
    cat("ERROR :", conditionMessage(e), "\n")
  })
}
file_name <- paste0("Resus/Example1b/P1_omega_W", l, ".Rdata")
save(omega_W, file = file_name)
}


N = 100
t0 = proc.time()

cl <- makeCluster(20)
registerDoParallel(cl)
registerDoSNOW(cl)
pb <- txtProgressBar(max = N, style = 3)
progress <- function(n)
  setTxtProgressBar(pb, n)
opts <- list(progress = progress)

foreach (
  l = 1:N,
  .options.snow = opts,
  .packages = c('np', 'MFSIS', 'reticulate')
) %dopar% {
  Calculation(l)
}

stopCluster(cl)
tall = proc.time() - t0
save(tall, file = 'Resus/Example1b/tall_P1.Rdata')