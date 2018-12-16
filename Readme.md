## TODO now
make a lstm n>1 and aggregation n>1 model where the slope of the slopes (or last two slopes of a ma) is used to predict a new slope which is used on the ref value of the labels. check r2

create the matrix per column instread of all concatenated columns and then allow an additional
encoding step on the whole (lstm, features)-matrix per column before we concatenate all
features into the final matrix. this way we could also encode the frequencies of the 
frequencies inside of one window. as a nice side effect we would also solve first point
on the todo-later list   

## TODO later
* try to concatenate columns early, so stacking different dimensions is no problem
* fix location so all batches match the expected length without repeating values
* implement zipline for a strategy backtest: https://www.zipline.io/
* find options data: https://amp.reddit.com/r/options/comments/3gupe5/is_there_a_quantopian_for_options/  