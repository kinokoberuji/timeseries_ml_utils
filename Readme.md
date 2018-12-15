## TODO now
mock a lstm 1 aggregation 1 log return model using same seeded random numbers. check confusion Matrix

https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.RandomState.html


make a lstm n>1 aggregation model where the slope of the features is applied to the ref value of the labels. check r2


make a lstm n>1 and aggregation n>1 model where the slope of the slopes (or last two slopes of a ma) is used to predict a new slope which is used on the ref value of the labels. check r2

## TODO late
* try to concatenate columns early, so stacking different dimensions is no problem
* fix location so all batches match the expected length
* implement zipline for a strategy backtest: https://www.zipline.io/
* find options data: https://amp.reddit.com/r/options/comments/3gupe5/is_there_a_quantopian_for_options/  