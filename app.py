import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse


st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("shear_ANN_Model.h5")
    return model


def app():
    df = pd.read_csv('shear_df.csv')
    shear_strain = df['Shear Strain']
    shear_stress = df['Shear Stress']
    scalar = pickle.load(open('scaling (1).pkl', 'rb'))
    model = load_model()
    st.title("Shear Stress Strain Error Estimation")
    # take 3 input values from user
    st.write("Enter the values of constants")
    b = st.number_input("θ∞", min_value=10, max_value=100)
    a = st.number_input("θ0 (Greater than θ∞)",
                        min_value=b+1, max_value=1000)
    c = st.number_input("τc0", min_value=0, max_value=100)
    d = st.number_input("τs (Greater than τc0)",
                        min_value=max(c+1, 50), max_value=150)
    scaled_input = scalar.transform([[a, b, c, d]])
    calc_stress = c+(d-c)*(1-np.exp(-1*a*shear_strain/d))+b*shear_strain
    plt.plot(shear_strain, shear_stress, 'r', label='Experimental')
    plt.plot(shear_strain, calc_stress, 'b', label='Calculated')
    plt.xlabel('Strain')
    plt.ylabel('Stress')
    plt.legend()
    st.write("The graph of the stress strain curve is shown below")
    st.pyplot(plt)
    st.write("Actual MAE : ", mae(shear_stress, calc_stress))
    st.write("Predicted MAE by ANN: ", model.predict(scaled_input)[0][0])


if __name__ == '__main__':
    app()
