import tensorflow as tf
from online_batches import online_batches

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
       

    def predict(self, t, data_batches: online_batches):
        # Call predict method on the loaded model
        temp_X, temp_y, temp_scalar, solar_X, solar_y, solar_scalar, load_X, load_y, load_scalar = data_batches.get_online_training_data()
        temp_hat = self._predict(self.temp_model, t,temp_X, temp_y,self.austin_N, temp_scalar)
        solar_hat = self._predict(self.solar_model, t,solar_X, solar_y,self.austin_N, solar_scalar)
        load_hat = self._predict(self.load_model, t,load_X, load_y,self.load_N, load_scalar)
        # need to divide solar by 1000 to conver to K(units)
        return temp_hat,solar_hat/1000,load_hat
    
    def _predict(self,model, t, X_test, y_test, N, scalar):
        x = X_test[t].reshape(-1,self.T,N)  # a "new" input is available
        y_hat = model.predict_on_batch(x) # predict on the "new" input
        # averaged = np.mean(y_hat[:,3:6,:], axis=1)
        y = y_test[t].reshape(1,-1,1)   # a "new" label is available
        model.train_on_batch(x, y)  # runs a single gradient update 
        return scalar.inverse_transform(y_hat)
