## TODO now

add a new fit callback just retuning metrics after each batch
 
create the matrix per column instead of all concatenated columns and then allow an additional
encoding step on the whole (lstm, features)-matrix per column before we concatenate all
features into the final matrix. this way we could also encode the frequencies of the 
frequencies inside of one window. as a nice side effect we would also solve first point
on the todo-later list   

convert encoder-decoder function to an object where we encode, decode and eventually
already transform the (lstm, feature) matrix as mentioned above
 
## TODO later
* try to concatenate columns early, so stacking different dimensions is no problem
* enable multiple assets as multiple data frames 
* fix location so all batches match the expected length without repeating values
* implement zipline for a strategy backtest: https://www.zipline.io/
* find options data: https://amp.reddit.com/r/options/comments/3gupe5/is_there_a_quantopian_for_options/  


## Network Ideas
1 take all available features normalized to its reference value and encode all labels as 
regression line.
  then try to predict the slope
  the direction of the slope
