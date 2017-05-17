
setwd("D:/Rdata/deep_learning/dat")

dat_train<- read.csv("D:/Rdata/Data/train.csv")

dat_train1<-dat_train[1:1000 ,]
dat_validation<-dat_train[1001:2000 ,]
dat_test<-dat_train[2001:8000 ,]


h2oactivity_train <- as.h2o(dat_train1,destination_frame = "dat_train_1")
h2oactivity_validation <- as.h2o(dat_validation,destination_frame = "dat_validation_1")
h2oactivity_test <- as.h2o(dat_test,destination_frame = "dat_test_1")

h2oactivity_train$label <- as.factor(h2oactivity_train$label)
h2oactivity_validation$label<- as.factor(h2oactivity_validation$label)
h2oactivity_test$label <- as.factor(h2oactivity_test$label)

h2oactivity_test_x <-h2oactivity_test[,-1]
h2oactivity_test_y<-h2oactivity_test[,1]

y <- "label"
x <- setdiff(names(dat_train), y)









m3 <- h2o.deeplearning(
  model_id="dl_model_tuned",
  training_frame=h2oactivity_train,
  validation_frame=h2oactivity_validation,
  x=x,
  y=y,
  overwrite_with_best_model=F, ## Return the final model after 10 epochs, even if not the best
  hidden=c(128,128,128), ## more hidden layers -> more complex interactions
  epochs=30, ## to keep it short enough
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025, ## don't score more than 2.5% of the wall time
  adaptive_rate=F, ## manually tuned learning rate
  rate=0.01,
  rate_annealing=2e-6,
  momentum_start=0.2, ## manually tuned momentum
  momentum_stable=0.4,
  momentum_ramp=1e7,
  l1=1e-5, ## add some L1/L2 regularization
  l2=1e-5,
  nfolds=5,
  max_w2=10 ,## helps stability for Rectifier
  variable_importances=T ,
  stopping_rounds=2,
  stopping_metric="misclassification", ## could be "MSE","logloss","r2"
  stopping_tolerance=0.01
)




summary(m3)

plot(m3)






h2o.performance(m3, train=T) ## sampled training data (from model building)
h2o.performance(m3, valid=T) ## sampled validation data (from model building)
h2o.performance(m3, newdata=h2oactivity_train) ## full training data
h2o.performance(m3, newdata=h2oactivity_validation) ## full validation data
h2o.performance(m3, newdata=h2oactivity_test_x) 




pred <- h2o.predict(m3, h2oactivity_test_x)

h2oactivity_test$Accuracy <- pred$predict == h2oactivity_test$label
1-mean(h2oactivity_test$Accuracy)

