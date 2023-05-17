import tensorflow as tf


class ml_model:
    
    def __init__(self) -> None:
        self.solar_model = tf.keras.models.load_model("saved_model/temp_Austin_solar_lstm3/")
        self.temp_model = tf.keras.models.load_model("saved_model/temp_Austin_temp_lstm3/")
        self.load_model = None
        # split the data into inputs and targets
        freq = 30          # 1 point every 15 min
        step = int(60/freq)    # number of points in an hour
        past = 6*24*step    # will be using last 7 days of data
        future = 1*24*step  # to predict next 1 day
        self.scaling_const = 1000

        # I want to use a T-days window of input data for predicting target_class
        # It means I need to prepend (T-1) last train records to the 1st test window
        self.T = past  # my choice of the timesteps window
        self.pred_T = future
        self.austin_N = 6   # number of features
        self.load_N = 1     # number of load features
        self.LR = 1e-3      # learning rate of the gradient descent
       

    def predict(self, t, temp_X, temp_y, solar_X, solar_y, load_X, load_y):
        # Call predict method on the loaded model
        temp_hat = self._predict(t,temp_X, temp_y, self.temp_model, self.austin_N)
        solar_hat = self._predict(t,solar_X, solar_y, self.solar_model, self.austin_N)
        load_hat = self._predict(t,load_X, load_y, self.load_model, self.load_N)
        return temp_hat,solar_hat,load_hat
    
    def _predict(self, t, X_test, y_test, model, N):
        x = X_test[t].reshape(-1,self.T,N)  # a "new" input is available
        y_hat = model.predict_on_batch(x) # predict on the "new" input
        # averaged = np.mean(y_hat[:,3:6,:], axis=1)
        y = y_test[t].reshape(1,-1,1)   # a "new" label is available
        model.train_on_batch(x, y)  # runs a single gradient update 
        return y_hat
