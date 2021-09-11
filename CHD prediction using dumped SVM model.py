import pandas as pd
import joblib

my_data = {'age':[63],
          'totChol':[205],
          'sysBP':[138],
          'diaBP':[120],
          'BMI':[33],
          'heartRate': [85],
          'glucose': [98]}

my_data = pd.DataFrame(my_data, columns=['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'])

filename = 'final_model.sav'
loaded_model = joblib.load(filename)
prediction = loaded_model.predict(my_data)
prediction
